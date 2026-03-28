const std = @import("std");
const math = std.math;

const zg = @import("../zigrad.zig");
const NDTensor = zg.NDTensor;
const settings = zg.settings;
const opspec = zg.opspec;

const Error = zg.device.Error || std.mem.Allocator.Error;
const OptimTag = enum { sgd, adam };
const UpdateCallback = *const fn (*anyopaque, *zg.Graph.Node) Error!void;
const StepCallback = *const fn (*anyopaque) Error!void;

/// ParamEntry is a closure to support
/// generic type optimization.
const ParamEntry = struct {
    node_ptr: *zg.Graph.Node,
    upd_call: UpdateCallback,
};

/// Every optimizer backend will have a ParamList that
/// tracks objects that the Optimizer has attached to.
const ParamList = std.array_list.Managed(ParamEntry);

/// Generic interface for handling optimizers. You can think
/// of this as being similar to an Allocator interface. Each
/// optimizer backend will have an `optimzer` function that
/// returns an instance of this class.
pub const Optimizer = struct {
    ptr: *anyopaque,
    tag: OptimTag,
    params: *ParamList,
    begin_step: ?StepCallback = null,

    pub fn attach(self: Optimizer, object: anytype) !void {
        comptime std.debug.assert(@typeInfo(@TypeOf(object)) == .pointer);
        const Object = std.meta.Child(@TypeOf(object));

        if (comptime !@hasField(Object, "node"))
            @compileError("Object does not have a 'node' field: " ++ @typeName(Object));

        try self.params.append(.{
            .node_ptr = &object.node,
            .upd_call = update_selector(self.tag, Object),
        });
    }

    pub fn step(self: Optimizer) Error!void {
        if (self.params.items.len == 0) return;
        if (self.begin_step) |begin_step| {
            try begin_step(self.ptr);
        }
        for (self.params.items) |entry| {
            try entry.upd_call(self.ptr, entry.node_ptr);
        }
    }

    fn update_selector(tag: OptimTag, Param: type) UpdateCallback {
        return switch (tag) {
            .sgd => update_wrapper(SGD, Param),
            .adam => update_wrapper(Adam, Param),
        };
    }

    fn update_wrapper(Optim: type, Param: type) UpdateCallback {
        return struct {
            pub fn update(ctx: *anyopaque, node: *zg.Graph.Node) Error!void {
                const optim: *Optim = @ptrCast(@alignCast(ctx));
                const param: *Param = node.upcast(Param);
                try optim.update(param);
            }
        }.update;
    }
};

pub const SGD = struct {
    params: ParamList,
    grad_clip_enabled: bool,
    grad_clip_max_norm: f32,
    grad_clip_delta: f32,
    lr: f64,

    pub fn init(allocator: std.mem.Allocator, opts: struct {
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,
        lr: f64,
    }) SGD {
        return .{
            .params = ParamList.init(allocator),
            .grad_clip_enabled = opts.grad_clip_enabled,
            .grad_clip_max_norm = opts.grad_clip_max_norm,
            .grad_clip_delta = opts.grad_clip_delta,
            .lr = opts.lr,
        };
    }

    pub fn deinit(self: *SGD) void {
        self.params.deinit();
    }

    pub fn optimizer(self: *SGD) Optimizer {
        return .{ .ptr = self, .tag = .sgd, .params = &self.params, .begin_step = null };
    }

    pub fn update(self: *SGD, param: anytype) Error!void {
        const Param = std.meta.Child(@TypeOf(param));

        const nlr = -@as(Param.ValueType, @floatCast(self.lr));

        switch (comptime Param.Category) {
            .dense => {
                if (self.grad_clip_enabled) param._clip_grad_norm(.{
                    .max_norm = self.grad_clip_max_norm,
                    .delta = self.grad_clip_delta,
                });
                // I suppose the idiomatic way would be to use the method
                // for (params) |param| param.data._axpy(param.grad.?, nlr, param.device);
                // But, can use direct access to skip the shape checks
                param.device.dispatch(opspec.axpy(Param.ValueType){
                    .x = param.assume_grad_data(),
                    .y = param.get_data(),
                    .alpha = &nlr,
                });
            },
            else => @compileError("Unimplemented: SGD for " ++ @typeName(Param)),
        }
    }
};

pub const Adam = struct {
    const MapEntry = struct { m: []u8, v: []u8, device: zg.DeviceReference };
    const ParamMap = std.AutoArrayHashMap(usize, MapEntry);

    params: ParamList,
    map: ParamMap,

    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    grad_clip_enabled: bool = settings.grad_clip_enabled,
    grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
    grad_clip_delta: f32 = settings.grad_clip_delta,
    t: usize,
    step_size: f64,

    pub fn init(allocator: std.mem.Allocator, opts: struct {
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        grad_clip_enabled: bool = settings.grad_clip_enabled,
        grad_clip_max_norm: f32 = settings.grad_clip_max_norm,
        grad_clip_delta: f32 = settings.grad_clip_delta,
    }) Adam {
        return .{
            .params = ParamList.init(allocator),
            .map = ParamMap.init(allocator),
            .lr = opts.lr,
            .beta1 = opts.beta1,
            .beta2 = opts.beta2,
            .epsilon = opts.epsilon,
            .grad_clip_enabled = opts.grad_clip_enabled,
            .grad_clip_max_norm = opts.grad_clip_max_norm,
            .grad_clip_delta = opts.grad_clip_delta,
            .t = 0,
            .step_size = 0,
        };
    }

    pub fn deinit(self: *Adam) void {
        self.params.deinit();
        for (self.map.values()) |entry| {
            entry.device.mem_free(entry.m);
            entry.device.mem_free(entry.v);
        }
        self.map.deinit();
    }

    pub fn optimizer(self: *Adam) Optimizer {
        return .{
            .ptr = self,
            .tag = .adam,
            .params = &self.params,
            .begin_step = beginStepCallback,
        };
    }

    fn beginStepCallback(ctx: *anyopaque) Error!void {
        const self: *Adam = @ptrCast(@alignCast(ctx));
        try self.beginStep();
    }

    fn beginStep(self: *Adam) Error!void {
        self.t += 1;
        const t_f: f64 = @floatFromInt(self.t);
        self.step_size = self.lr * math.sqrt(1 - math.pow(f64, self.beta2, t_f)) / (1 - math.pow(f64, self.beta1, t_f));
    }

    pub fn update(self: *Adam, param: anytype) Error!void {
        const Param = std.meta.Child(@TypeOf(param));
        const T = Param.ValueType;

        const beta1: T = @floatCast(self.beta1);
        const beta2: T = @floatCast(self.beta2);
        const one_minus_beta1: T = @floatCast(1 - self.beta1);
        const one_minus_beta2: T = @floatCast(1 - self.beta2);
        const step_size: T = @floatCast(self.step_size);
        const epsilon: T = @floatCast(self.epsilon);

        switch (comptime Param.Category) {
            .dense => {
                if (self.grad_clip_enabled) param._clip_grad_norm(.{
                    .max_norm = self.grad_clip_max_norm,
                    .delta = self.grad_clip_delta,
                });

                const param_size = param.get_size();

                const m, const v = blk: { // initialize m and v
                    const gop = try self.map.getOrPut(@intFromPtr(param));
                    if (!gop.found_existing) {
                        const m_bytes = try param.device.mem_alloc(u8, param_size * @sizeOf(T));
                        const v_bytes = try param.device.mem_alloc(u8, param_size * @sizeOf(T));
                        param.device.mem_fill(u8, m_bytes, 0);
                        param.device.mem_fill(u8, v_bytes, 0);
                        gop.value_ptr.* = .{
                            .m = m_bytes,
                            .v = v_bytes,
                            .device = param.device,
                        };
                    }
                    break :blk .{
                        std.mem.bytesAsSlice(T, gop.value_ptr.m),
                        std.mem.bytesAsSlice(T, gop.value_ptr.v),
                    };
                };

                const p_data = param.get_data();
                const p_grad = param.assume_grad_data();

                param.device.dispatch(opspec.adam(T){
                    .param = p_data,
                    .grad = p_grad,
                    .m = m,
                    .v = v,
                    .beta1 = beta1,
                    .beta2 = beta2,
                    .one_minus_beta1 = one_minus_beta1,
                    .one_minus_beta2 = one_minus_beta2,
                    .step_size = step_size,
                    .epsilon = epsilon,
                });
            },
            else => @compileError("Unimplemented: Adam for " ++ @typeName(Param)),
        }
    }
};

test "adam uses one timestep per optimizer step across attached parameters" {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    zg.global_graph_init(std.testing.allocator, .{});
    defer zg.global_graph_deinit();

    var adam = Adam.init(std.testing.allocator, .{
        .lr = 0.1,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .grad_clip_enabled = false,
    });
    defer adam.deinit();

    const optimizer = adam.optimizer();
    const param_a = try NDTensor(f32).from_slice(device, &.{1.0}, &.{1}, .{ .requires_grad = true });
    defer param_a.deinit();
    const param_b = try NDTensor(f32).from_slice(device, &.{2.0}, &.{1}, .{ .requires_grad = true });
    defer param_b.deinit();

    try optimizer.attach(param_a);
    try optimizer.attach(param_b);

    try param_a.setup_grad(0.5);
    try param_b.setup_grad(0.5);

    try optimizer.step();

    try std.testing.expectApproxEqAbs(@as(f32, 0.9), param_a.get_data()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.9), param_b.get_data()[0], 1e-5);
    try std.testing.expectEqual(@as(usize, 1), adam.t);

    try optimizer.step();

    try std.testing.expectApproxEqAbs(@as(f32, 0.8), param_a.get_data()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.8), param_b.get_data()[0], 1e-5);
    try std.testing.expectEqual(@as(usize, 2), adam.t);
}

test {
    std.testing.refAllDeclsRecursive(@This());
}
