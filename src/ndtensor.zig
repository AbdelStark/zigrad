const std = @import("std");
const zg = @import("zigrad.zig");
const settings = zg.settings;
const DeviceReference = zg.DeviceReference;
const backend = zg.backend;
const opspec = zg.opspec;
const utils = @import("ndtensor/utils.zig");

const Graph = zg.Graph;
const Node = Graph.Node;
const DeviceData = zg.device.DeviceData;

pub const TensorOpts = @import("ndtensor/utils.zig").TensorOpts;
pub const Op = @import("ndtensor/utils.zig").Op;

const ndarray = @import("ndarray.zig");
const Range = ndarray.Range;
const Shape = ndarray.Shape;
const NDArray = ndarray.NDArray;
const BoundedArray = @import("utils/bounded_array.zig").BoundedArray;
const log = zg.logging.scoped(.zg_ndtensor);

pub const MaxAlongOptions = struct {
    dim: usize,
    keep_dims: bool = false,
    return_indices: bool = false,
    // TODO: Add checkpoint support later
    // checkpoint: bool = false,
};

pub fn NDTensor(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const ValueType = T;
        pub const DataType = NDArray(T);
        pub const Category = zg.Category.dense;
        pub const Status = enum { owned, view };
        /// Core NDArray that holds the values and shape.
        /// Use this member directly when you want to perform
        /// ops that will not be tracked by the graph.
        data: DataType,
        /// The gradient calculated by calling "backward".
        /// Gradients are lazily initialized.
        grad: ?DataType = null,
        /// The device field is a reference to a stateful
        /// object that provides memory and compute resources.
        device: DeviceReference,
        /// Denotes if a tensor is owned or viewed. Owned tensors
        /// are freed when calling deinit.
        status: Status = .owned,
        /// Intrusive node that hooks up the NDTensor class to
        /// a zigrad computation graph.
        node: Node,
        /// Optional op tag - TODO: do we need to
        /// continue to support this?
        op: ?Op = null,

        /// Shape is allocated. COM.
        pub fn empty(device: DeviceReference, shape: []const usize, opts: TensorOpts) !*Self {
            var graph = opts.graph orelse zg.global_graph_get();
            const self = try graph.builder.create_node(Self);
            errdefer graph.builder.destroy_node(self);

            self.* = .{
                .data = try DataType.empty(shape, device),
                .device = device,
                .node = .init(Self, &graph.builder, null, opts.label, .{
                    .requires_grad = opts.requires_grad,
                    .acquired = opts.acquired,
                    .attached = opts.attached,
                }),
            };

            try self.captureStandalone("source");
            return self;
        }

        /// Transfers a host-slice to device memory. Helpful for constructing tests from static arrays.
        pub fn from_slice(device: DeviceReference, values: []const T, shape: ?[]const usize, opts: TensorOpts) !*Self {
            const self = try Self.empty(device, shape orelse &.{values.len}, opts);
            self.device.mem_transfer(T, values, self.get_data(), .HtoD);
            return self;
        }

        pub fn zeros(device: DeviceReference, shape: []const usize, opts: TensorOpts) !*Self {
            const self = try Self.empty(device, shape, opts);
            self.fill(0);
            return self;
        }

        pub fn ones(device: DeviceReference, shape: []const usize, opts: TensorOpts) !*Self {
            const self = try Self.empty(device, shape, opts);
            self.fill(1);
            return self;
        }

        pub fn random(device: DeviceReference, shape: []const usize, rt: zg.RandType, opts: TensorOpts) !*Self {
            const self = try Self.empty(device, shape, opts);
            device.mem_random(T, self.get_data(), rt, zg.random);
            return self;
        }

        pub fn sequence(device: DeviceReference, start: T, step: T, shape: []const usize, opts: TensorOpts) !*Self {
            const self = try Self.empty(device, shape, opts);
            self.device.mem_sequence(T, self.get_data(), start, step);
            return self;
        }

        ///////////////////////////////////////////////////////
        // Flag Helpers ///////////////////////////////////////

        pub fn attached(self: *const Self) bool {
            return self.node.attached();
        }

        pub fn attach(self: *Self) void {
            self.node.flags.set(.attached, true);
        }

        pub fn detach(self: *Self) void {
            self.node.flags.set(.attached, false);
        }

        pub fn requires_grad(self: *const Self) bool {
            return self.node.requires_grad();
        }

        pub fn enable_grad(self: *Self) void {
            self.node.flags.set(.requires_grad, true);
        }

        pub fn disable_grad(self: *Self) void {
            self.node.flags.set(.requires_grad, false);
        }

        pub fn acquired(self: *const Self) bool {
            return self.node.acquired();
        }

        pub fn acquire(self: *Self) void {
            self.node.flags.set(.acquired, true);
        }

        pub fn release(self: *Self) void {
            self.node.flags.set(.acquired, false);
        }

        /// This function should be checked by the user to see
        /// if they should deinitialize the memory on a forward
        /// pass. If the tensor requires a gradient, then it must
        /// exist for the backward pass to function. Likewise,
        /// acquired memory should never be deinitialized without
        /// first calling "release" .
        pub fn should_deinit(self: *const Self) bool {
            return !(self.acquired() or self.requires_grad());
        }

        pub fn backward(self: *Self) !void {
            std.debug.assert(zg.runtime.grad_enabled);
            const graph = self.node.gb.promote();
            _ = try self.ensure_grad(1);
            try graph.backward(&self.node);
        }

        pub fn teardown(self: *Self) !void {
            const graph = self.node.gb.promote();
            graph.teardown(&self.node);
        }

        ///////////////////////////////////////////////////////
        // Tensor Component Helpers ///////////////////////////

        pub fn get_shape(self: *const Self) []const usize {
            return self.data.shape.slice();
        }

        pub fn get_size(self: *const Self) usize {
            return self.data.size();
        }

        pub fn get_strides(self: *const Self) Shape.Strides {
            return self.data.shape.strides();
        }

        pub fn get_data(self: *const Self) []T {
            return self.data.get_data();
        }

        pub fn copy_to_host(self: *const Self, dst: []T) !void {
            if (dst.len != self.get_size()) return error.InvalidHostCopySize;
            try self.captureMaterialization(.host_read);
            zg.lazy.flushIfDeferred();

            if (self.device.is_host()) {
                @memcpy(dst, self.get_data());
                return;
            }

            self.device.sync();
            self.device.mem_transfer(T, self.get_data(), dst, .DtoH);
            self.device.sync();
        }

        pub fn to_host_owned(self: *const Self, allocator: std.mem.Allocator) ![]T {
            const host = try allocator.alloc(T, self.get_size());
            errdefer allocator.free(host);
            try self.copy_to_host(host);
            return host;
        }

        pub fn realize(self: *Self) !*Self {
            try self.captureMaterialization(.explicit);
            zg.lazy.flushIfDeferred();
            self.device.sync();
            return self;
        }

        pub fn get_dim(self: *const Self, i: usize) usize {
            return self.data.shape.get(i);
        }

        pub fn cast(self: *Self, K: type) !*NDTensor(K) {
            _ = self;
            @compileError("Not implemented");
        }

        /// In-place unsqueeze, does not provide a backward.
        pub fn _unsqueeze(self: *Self) void {
            self.data.shape._unsqueeze();
            if (self.grad) |*g| g.shape._unsqueeze();
        }

        /// In-place squeeze, does not provide a backward.
        pub fn _squeeze(self: *Self) void {
            self.data.shape._squeeze();
            if (self.grad) |*g| g.shape._squeeze();
        }

        pub fn setup_grad(self: *Self, fill_value: ?T) !void {
            if (self.grad == null) {
                self.grad = try DataType.empty(self.get_shape(), self.device);
            }
            return self.assume_grad().fill(fill_value orelse return, self.device);
        }

        pub fn assume_grad(self: *Self) *DataType {
            if (self.grad) |*grd| {
                return grd;
            } else {
                @branchHint(.cold);
                @panic("no gradient");
            }
        }

        pub fn assume_grad_data(self: *Self) []T {
            return self.assume_grad().get_data();
        }

        // This function can allocate a gradient if one is not present.
        pub fn ensure_grad(self: *Self, fill_value: ?T) !*DataType {
            if (self.grad) |*grd| {
                return grd;
            } else {
                try self.setup_grad(fill_value);
                return self.assume_grad();
            }
        }

        // This function can allocate a gradient if one is not present.
        pub fn ensure_grad_data(self: *Self, fill_value: ?T) ![]T {
            const grd = try self.ensure_grad(fill_value);
            return grd.get_data();
        }

        pub fn get_label(self: *const Self) ?[]const u8 {
            return self.node.get_label();
        }

        pub fn set_label(self: *Self, new_label: []const u8) void {
            self.node.set_label(new_label);
        }

        pub fn CreateDependentOpts(BwdCallback: type) type {
            return struct {
                data: DataType,
                grad: ?DataType = null,
                gb: *Graph.Builder,
                children: []const *Node,
                callback: BwdCallback,
                device: DeviceReference,
                status: Status = .owned,
                label: ?[]const u8 = null,
                op: ?Op = null,
                capture_name: ?[]const u8 = null,
                capture_attributes: []const zg.lazy.OpAttribute = &.{},
            };
        }

        pub fn create_dependent(BwdClosureType: type, opts: CreateDependentOpts(BwdClosureType)) !*Self {
            const req_grad: bool = for (opts.children) |child| {
                if (child.requires_grad()) break true;
            } else false;

            if (req_grad) for (opts.children) |child|
                child.flags.set(.grad_operand, true);

            const self = try opts.gb.create_node(Self);
            errdefer opts.gb.destroy_node(self);

            const bwd_ctx: ?Node.BackwardContext = if (req_grad)
                try .init(Self, BwdClosureType, opts.gb.allocator, opts.callback, opts.children)
            else
                null;

            self.* = Self{
                .data = opts.data,
                .grad = opts.grad,
                .device = opts.device,
                .status = opts.status,
                .op = opts.op,
                .node = .init(Self, opts.gb, bwd_ctx, opts.label, .{
                    .requires_grad = req_grad,
                    .acquired = false,
                    .attached = true,
                }),
            };

            try self.captureTensor(
                opts.capture_name orelse if (opts.op) |op| @tagName(op) else "op",
                opts.children,
                opts.capture_attributes,
            );
            return self;
        }

        pub fn prepend_dependent(BwdClosureType: type, self: *Self, opts: struct {
            children: []const *Node,
            callback: BwdClosureType,
        }) !void {
            const req_grad: bool = for (opts.children) |child| {
                if (child.requires_grad()) break true;
            } else self.requires_grad();

            if (req_grad) for (opts.children) |child|
                child.flags.set(.grad_operand, true);

            if (req_grad) {
                const new_ctx: Node.BackwardContext = try .init(
                    Self,
                    BwdClosureType,
                    self.node.gb.allocator,
                    opts.callback,
                    opts.children,
                );

                if (self.node.callbacks.bwd) |*old_ctx| {
                    try old_ctx.prepend(new_ctx, self.node.gb.allocator);
                } else {
                    self.node.callbacks.bwd = new_ctx;
                }
                // I'm using wrapping add on the off-chance that someone
                // overflows the byte - it would still be a mismatch on
                // the backward pass whereas saturation wouldn't.
                // Since this is effectively a modulus, the user could
                // setup an extremely odd situation (with more than 256 inplace
                // ops on a single tensor) that would cause a false pass,
                // but I don't see the need to address such edge cases.
                // While this only checked in debug mode, its free to track so the
                //   user (or we) can verify the graph once or as-needed if desired.
                self.node.version +%= 1;
            }
        }

        /// Free all device related memory associated with a tensor. The graph owns
        /// the tensor instance, so reset or deinit should be called on the owning
        /// graph to destroy the instance itself.
        pub fn deinit(self: *Self) void {
            if (self.acquired())
                @panic("Attempt to deinit an acquired tensor.");

            if (!self.node.active())
                return;

            if (self.status == .owned)
                self.data.deinit(self.device);

            // Viewing tensors have gradients
            // independent of origin tensors.
            if (self.grad != null) {
                self.grad.?.deinit(self.device);
                self.grad = null;
            }

            self.node.deactivate();

            self.node.gb.destroy_node(self);
        }

        /// Checks to see if a node is acquired or is the operand of a node that
        /// requires a gradient. If neither are true, the tensor is freed. Usually
        /// called in forward contexts when working with a mixed gradient requirements
        /// and view tensors.
        pub fn soft_deinit(self: *Self) void {
            if (!(self.acquired() or self.node.flags.get(.grad_operand)))
                self.deinit();
        }

        fn to_device_impl(
            src: []const T,
            dst: []T,
            src_device: DeviceReference,
            dst_device: DeviceReference,
        ) !void {
            if (comptime @typeInfo(@TypeOf(src_device)) != .@"struct")
                return;

            // currently only implemented for a single aux device.
            std.debug.assert(!src_device.is_compatible(dst_device));

            if (src_device.is_host()) {
                dst_device.sync();
                dst_device.mem_transfer(T, src, dst, .HtoD);
            } else {
                src_device.sync();
                src_device.mem_transfer(T, src, dst, .DtoH);
            }
        }

        pub fn _to_device(self: *Self, device: DeviceReference) !void {
            if (self.device.is_compatible(device))
                return;

            const data = try device.mem_cache_alloc(T, self.get_size());
            try to_device_impl(self.get_data(), data, self.device, device);
            self.data.deinit(self.device);
            self.data.data = data;
            self.device = device;
        }

        pub fn to_device(self: *Self, device: DeviceReference) !*Self {
            if (self.device.is_compatible(device))
                return self;

            const ToDeviceBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    try to_device_impl(
                        y.assume_grad_data(),
                        try x.ensure_grad_data(0),
                        y.device,
                        x.device,
                    );
                }
            };

            const data = try device.mem_cache_alloc(T, self.get_size());
            errdefer device.mem_cache_free(data);

            try to_device_impl(self.get_data(), data.raw, self.device, device);

            return try create_dependent(ToDeviceBwd, .{
                .data = .{
                    .data = data,
                    .shape = self.data.shape,
                },
                .children = &.{&self.node},
                .device = device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .TRANSFER,
                .capture_attributes = &.{
                    .{ .key = "source_device", .value = .{ .string = if (self.device.is_host()) "host" else "cuda" } },
                    .{ .key = "target_device", .value = .{ .string = if (device.is_host()) "host" else "cuda" } },
                },
            });
        }

        /// Backing data and grad are copied, uses the same graph body
        /// for AD node, requires_grad status is retained.
        ///
        /// Other metadata is not retained and is reset to defaults,
        /// such as attached and acquired statuses.
        pub fn clone(self: *const Self) !*Self {
            const result = try self.node.gb.create_node(Self);
            errdefer self.node.gb.destroy_node(result);

            var data = try self.data.copy(self.device);
            errdefer data.deinit(self.device);
            const default_opts = TensorOpts{};
            result.* = Self{
                .data = data,
                .grad = if (self.grad) |*g| try g.copy(self.device) else null,
                .device = self.device,
                .node = .init(Self, self.node.gb, null, null, .{
                    .requires_grad = self.requires_grad(),
                    .acquired = default_opts.acquired,
                    .attached = default_opts.attached,
                }),
            };

            try result.captureStandalone("clone");
            return result;
        }

        pub fn log_shape(self: *const Self, comptime msg: ?[]const u8) void {
            log.debug("{s}{s} data shape: {any} grad shape: {?any}", .{
                if (msg) |n| n else "",
                if (self.get_label()) |l| l else "",
                self.data.shape.slice(),
                if (self.grad) |g| g.shape.slice() else null,
            });
        }

        /// In-place, no backward.
        pub fn _reshape(self: *Self, shape: []const usize) void {
            self.data._reshape(shape);
            if (self.grad) |*g| g._reshape(shape);
            if (self.requires_grad() or self.node.flags.get(.grad_operand)) self.node.version +%= 1;
        }

        /// Copies. COM.
        pub fn reshape(self: *Self, new_shape: []const usize) !*Self {
            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "new_shape", .value = .{ .usize_list = new_shape } },
            };
            const ReshapeBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    const x_grad = try x.ensure_grad(0);
                    y.assume_grad()._reshape(x.data.shape.slice());
                    try x_grad._add(y.assume_grad().*, y.device);
                }
            };

            var result = try self.data.copy(self.device);
            result._reshape(new_shape);

            return try create_dependent(ReshapeBwd, .{
                .data = result,
                .children = &.{&self.node},
                .gb = self.node.gb,
                .device = self.device,
                .callback = .{},
                .op = .RESHAPE,
                .capture_attributes = &capture_attributes,
            });
        }

        /// Copies. COM.
        pub fn transpose(self: *Self) !*Self {
            const permutation = [_]usize{ 1, 0 };
            const TransposeBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.transpose(T){
                        .A = y.assume_grad_data(),
                        .B = try x.ensure_grad_data(0),
                        .m = y.get_dim(0),
                        .n = y.get_dim(1),
                        .alpha = 1.0,
                    });
                }
            };

            return try create_dependent(TransposeBwd, .{
                .data = try self.data.transpose(self.device),
                .gb = self.node.gb,
                .children = &.{&self.node},
                .device = self.device,
                .callback = .{},
                .op = .TRANSPOSE,
                .capture_attributes = &.{
                    .{ .key = "permutation", .value = .{ .usize_list = &permutation } },
                },
            });
        }

        pub fn fill(self: *const Self, val: T) void {
            self.data.fill(val, self.device);
        }

        /// Standard value-getter. Try to avoid using this when
        /// working with device memory because it's expensive.
        /// Get is not a gradient tracked operation.
        pub fn get(self: *const Self, idx: usize) T {
            var tmp: [1]T = undefined;
            self.device.mem_transfer(T, self.data.data.raw[idx .. idx + 1], tmp[0..], .DtoH);
            return tmp[0];
        }

        /// Standard value-setter. Try to avoid using this when
        /// working with device memory because it's expensive.
        /// Set is not a gradient tracked operation.
        pub fn set(self: *const Self, idx: usize, value: T) void {
            const tmp: [1]T = @splat(value);
            self.device.mem_transfer(T, tmp[0..], self.data.data[idx .. idx + 1], .HtoD);
        }

        /// Tensor value-setter.
        ///
        /// x.set_offset(n, y) where y.get_size() -> n: copies y into x[offset..offset + n]
        pub fn set_offset(dst: *Self, offset: usize, src: *const Self) void {
            std.debug.assert(src.get_size() <= dst.get_size());
            const end = offset + src.get_size();
            const src_data = src.get_data();
            const dst_data = dst.get_data()[offset..end];

            if (src.device.is_compatible(dst.device)) {
                dst.device.mem_copy(T, src_data, dst_data);
            } else if (dst.device.is_host()) {
                src.device.mem_transfer(T, src_data, dst_data, .DtoH);
            } else {
                dst.device.mem_transfer(T, src_data, dst_data, .HtoD);
            }
        }

        /// Tensor value-getter.
        ///
        /// x.get_offset(n, y) where y.get_size() -> n: copies x[offset..offset + n] into y
        pub fn get_offset(src: *const Self, offset: usize, dst: *Self) void {
            std.debug.assert(src.get_size() >= dst.get_size());
            const end = offset + dst.get_size();
            const src_data = src.get_data()[offset..end];
            const dst_data = dst.get_data();

            if (src.device.is_compatible(dst.device)) {
                dst.device.mem_copy(T, src_data, dst_data);
            } else if (dst.device.is_host()) {
                src.device.mem_transfer(T, src_data, dst_data, .HtoD);
            } else {
                dst.device.mem_transfer(T, src_data, dst_data, .DtoH);
            }
        }

        pub fn subset(self: *Self, steps: []const i64, status: Status) !*Self {
            std.debug.assert(self.data.shape.len >= steps.len);

            const SubsetBwds = struct {
                start: usize,
                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    const x_grad_data = (try x.ensure_grad_data(0))[ctx.start..][0..y.get_size()];
                    x.device.dispatch(opspec.add(T){
                        .x = x_grad_data,
                        .y = y.assume_grad_data(),
                        .z = x_grad_data,
                    });
                }
            };

            const strides = self.data.shape.strides();

            var start: usize = 0;
            var partial_size: usize = self.get_size();
            for (steps, 0..) |step, i| {
                std.debug.assert(@abs(step) < self.data.shape.get(i));

                if (step < 0) {
                    const total_steps = partial_size / strides.get(i);
                    const actual_step = total_steps - @abs(step);
                    start += actual_step * strides.get(i);
                } else {
                    start += @abs(step) * strides.get(i);
                }

                partial_size /= self.data.shape.get(i);
            }

            const tail = self.data.shape.tail(self.data.shape.len - steps.len);
            const size = Shape.slice_size(tail);

            const raw_data = switch (status) {
                .owned => try self.device.mem_cache_dupe(T, self.get_data()[start..][0..size]),
                .view => DeviceData(T){ .raw = self.get_data()[start..][0..size], .ctx = 0 },
            };
            errdefer if (status == .owned)
                self.device.mem_cache_free(raw_data);

            const tmp = try create_dependent(SubsetBwds, .{
                .data = .{
                    .data = raw_data,
                    .shape = Shape.init(tail),
                },
                .gb = self.node.gb,
                .children = &.{&self.node},
                .device = self.device,
                .status = status,
                .callback = .{ .start = start },
                .capture_name = "subset",
                .capture_attributes = &.{
                    .{ .key = "steps", .value = .{ .int_list = steps } },
                },
            });
            return tmp;
        }

        /// Create a tensor that shares underlying memory, but does not share
        /// shape or gradient.
        pub fn alias(self: *Self) !*Self {
            const AliasBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    const x_grad_data = try x.ensure_grad_data(0);
                    x.device.dispatch(opspec.add(T){
                        .x = x_grad_data,
                        .y = y.assume_grad_data(),
                        .z = x_grad_data,
                    });
                }
            };
            const tmp = try create_dependent(AliasBwd, .{
                .data = self.data,
                .gb = self.node.gb,
                .children = &.{&self.node},
                .device = self.device,
                .status = .view,
                .callback = .{},
                .capture_name = "alias",
            });
            return tmp;
        }

        pub fn print(self: *const Self) void {
            // self.print_to_writer(std.fs.File.stdout().deprecatedWriter());
            self.print_to_writer(std.fs.File.stderr().deprecatedWriter()) catch @panic("Failed to print tensor");
        }

        pub fn print_to_writer(self: *const Self, writer: anytype) !void {
            try writer.print("NDTensor<{},{?s}>", .{ T, if (self.op) |o| @tagName(o) else null });
            try writer.writeAll("{data: ");
            try self.data.print_to_writer(writer, self.device);
            if (self.grad) |g| {
                try writer.writeAll(" grad: ");
                try g.print_to_writer(writer, self.device);
            }
            try writer.print(", requires_grad={}", .{self.requires_grad()});
            if (self.get_label()) |l| {
                try writer.print(", label={s}", .{l});
            }
            try writer.writeAll("}\n");
        }

        pub const ClipOptions = struct {
            max_norm: f32 = settings.grad_clip_max_norm,
            delta: f32 = settings.grad_clip_delta,
        };

        pub fn _clip_grad_norm(self: *Self, opts: ClipOptions) void {
            self.assume_grad()._clip_norm(opts.max_norm, opts.delta, self.device);
        }

        /// Direct modification. Clamps the underlying data, as with all in place ops you must know what you are doing.
        /// This operation is not tracked in the computation graph.
        /// *Will not notify you of an improper gradient calculation.*
        pub fn _clamp(self: *Self, vmin: T, vmax: T) void {
            self.data._clamp(vmin, vmax, self.device);
        }

        /// Direct modification. Clamps the underlying grad, as with all in place ops you must know what you are doing.
        /// This operation is not tracked in the computation graph.
        /// *Will not notify you of an improper gradient calculation.*
        /// Grad must exist.
        pub fn _clamp_grad(self: *const Self, vmin: T, vmax: T) !void {
            (self.grad orelse return error.NoGradient)._clamp(vmin, vmax, self.device);
        }

        /// Clamp values to $[vmin, vmax]$
        pub fn clamp(self: *Self, vmin: T, vmax: T) !*Self {
            std.debug.assert(vmin <= vmax);

            const ClampBwd = struct {
                _min: T,
                _max: T,
                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.clamp_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y_g = y.assume_grad_data(),
                        .min = ctx._min,
                        .max = ctx._max,
                    });
                }
            };

            return create_dependent(ClampBwd, .{
                .data = try self.data.clamp(vmin, vmax, self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{ ._min = vmin, ._max = vmax },
                .op = .CLAMP,
            });
        }

        pub fn add_scalar(self: *Self, s: T) !*Self {
            const AddBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    const a = children.get_bwd_upcast(Self, 0) orelse return;
                    try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                }
            };
            return create_dependent(AddBwd, .{
                .data = try self.data.add_scalar(s, self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .ADD,
            });
        }

        pub fn sub_scalar(self: *Self, s: T) !*Self {
            return self.add_scalar(-s);
        }

        /// Element-wise addition. COM.
        pub fn add(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const AddBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    scope: {
                        const a = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(AddBwd, .{
                .data = try self.data.add(other.data, self.device),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .ADD,
            });
        }

        pub fn add_(self: *Self, other: *Self) !void {
            std.debug.assert(self.device.is_compatible(other.device));

            const InplaceAddBwd = struct {
                pub fn backward(b: *Self, children: *Node.Children) !void {
                    const a = children.get_bwd_upcast(Self, 0) orelse return;
                    try b.assume_grad().unbroadcast_(try a.ensure_grad(0), b.device, .{ .alpha = 1.0, .beta = 1.0 });
                }
            };

            try self.data.add_(&other.data, self.device);

            return prepend_dependent(InplaceAddBwd, other, .{
                .children = &.{&self.node},
                .callback = .{},
            });
        }

        pub fn _add(self: *Self, other: *Self) !void {
            return other.add_(self);
        }

        /// Element-wise subtraction. COM.
        pub fn sub(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const SubBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    scope: {
                        const a = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        try c.assume_grad().unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = -1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(SubBwd, .{
                .data = try self.data.sub(other.data, self.device),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .SUB,
            });
        }

        /// Element-wise multiplication. COM.
        pub fn mul(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const MulBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    scope: {
                        const a = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        const b = children.get_upcast(Self, 1);

                        var bc_grad = try b.data.mul(c.assume_grad().*, c.device);
                        defer bc_grad.deinit(c.device);

                        try bc_grad.unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        const a = children.get_upcast(Self, 0);

                        var ac_grad = try a.data.mul(c.assume_grad().*, c.device);
                        defer ac_grad.deinit(c.device);

                        try ac_grad.unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(MulBwd, .{
                .data = try self.data.mul(other.data, self.device),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .MUL,
            });
        }

        /// Element-wise division. COM.
        pub fn div(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const DivBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    scope: {
                        const a = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        const b = children.get_upcast(Self, 1);

                        var bc_grad = try c.assume_grad().div(b.data, c.device);
                        defer bc_grad.deinit(c.device);

                        try bc_grad.unbroadcast_(try a.ensure_grad(0), c.device, .{ .alpha = 1.0, .beta = 1.0 });
                    }
                    scope: {
                        const b = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        const a = children.get_upcast(Self, 0);

                        var ac_grad = blk: {
                            var b_grad_value = try c.assume_grad().mul(a.data, c.device);
                            defer b_grad_value.deinit(c.device);
                            var bsq = try b.data.mul(b.data, c.device);
                            defer bsq.deinit(c.device);
                            break :blk try b_grad_value.div(bsq, c.device);
                        };
                        defer ac_grad.deinit(c.device);

                        try ac_grad.unbroadcast_(try b.ensure_grad(0), c.device, .{ .alpha = -1.0, .beta = 1.0 });
                    }
                }
            };

            return create_dependent(DivBwd, .{
                .data = try self.data.div(other.data, self.device),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .DIV,
            });
        }

        /// Computes the maximum value of the tensor. Returns a scalar tensor. COM.
        pub fn max(self: *Self) !*Self {
            const MaxBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.max_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };

            return create_dependent(MaxBwd, .{
                .data = try self.data.max(self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .MAX,
            });
        }

        /// Element-wise exponential. COM.
        pub fn exp(self: *Self) !*Self {
            const ExpBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.exp_bwd(T){
                        .x_g = try x.ensure_grad_data(0),
                        .y = y.get_data(),
                        .y_g = y.assume_grad_data(),
                    });
                }
            };
            return create_dependent(ExpBwd, .{
                .data = try self.data.exp(self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .EXP,
            });
        }

        /// Element-wise pow.
        pub fn pow(self: *Self, exponent: T) !*Self {
            const PowBwd = struct {
                exp: T,

                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.pow_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .exp = ctx.exp,
                        .y_g = y.assume_grad_data(),
                        .eps = settings.eps,
                    });
                }
            };

            return create_dependent(PowBwd, .{
                .data = try self.data.pow(exponent, self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{ .exp = exponent },
                .op = .POW,
            });
        }

        /// Differentiable element-wise square root.
        pub fn sqrt(self: *Self) !*Self {
            const SqrtBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.sqrt_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y_g = y.assume_grad_data(),
                        .eps = settings.eps,
                    });
                }
            };

            return create_dependent(SqrtBwd, .{
                .data = try self.data.sqrt(self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .capture_name = "sqrt",
            });
        }

        /// Differentiable element-wise inverse square root.
        pub fn rsqrt(self: *Self) !*Self {
            const RSqrtBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    y.device.dispatch(opspec.rsqrt_bwd(T){
                        .x = x.get_data(),
                        .x_g = try x.ensure_grad_data(0),
                        .y_g = y.assume_grad_data(),
                        .eps = settings.eps,
                    });
                }
            };

            return create_dependent(RSqrtBwd, .{
                .data = try self.data.rsqrt(self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .capture_name = "rsqrt",
            });
        }

        const BmmOpts = struct {
            trans_a: bool = false,
            trans_b: bool = false,
            alpha: T = 1.0,
            beta: T = 0.0,
        };

        /// Matrix multiplication. COM.
        pub fn bmm(self: *Self, other: *Self, opts: BmmOpts) !*Self {
            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "trans_a", .value = .{ .bool = opts.trans_a } },
                .{ .key = "trans_b", .value = .{ .bool = opts.trans_b } },
                .{ .key = "alpha", .value = .{ .float = @floatCast(opts.alpha) } },
                .{ .key = "beta", .value = .{ .float = @floatCast(opts.beta) } },
            };
            return create_dependent(BmmAccBwd, .{
                .data = try self.data.bmm(other.data, self.device, .{
                    .trans_a = opts.trans_a,
                    .trans_b = opts.trans_b,
                    .alpha = opts.alpha,
                    .beta = opts.beta,
                }),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = Op.matmul_tag(opts.trans_a, opts.trans_b),
                .capture_attributes = &capture_attributes,
            });
        }

        pub fn bmm_acc_(self: *Self, other: *Self, out: *Self, opts: BmmOpts) !*Self {
            try self.data.bmm_acc_(other.data, &out.data, self.device, .{
                .trans_a = opts.trans_a,
                .trans_b = opts.trans_b,
                .alpha = opts.alpha,
                .beta = opts.beta,
            });

            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "trans_a", .value = .{ .bool = opts.trans_a } },
                .{ .key = "trans_b", .value = .{ .bool = opts.trans_b } },
                .{ .key = "alpha", .value = .{ .float = @floatCast(opts.alpha) } },
                .{ .key = "beta", .value = .{ .float = @floatCast(opts.beta) } },
            };
            return create_dependent(BmmAccBwd, .{
                .data = out.data,
                .grad = out.grad,
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = Op.matmul_tag(opts.trans_a, opts.trans_b),
                .capture_attributes = &capture_attributes,
            });
        }

        const BmmAccBwd = struct {
            pub fn backward(C: *Self, children: *Node.Children) !void {
                const op_tag = C.op orelse unreachable;
                const A = children.get_upcast(Self, 0);
                const B = children.get_upcast(Self, 1);

                if (children.get_bwd_upcast(Self, 0)) |_| {
                    const C_grad = C.assume_grad().*;
                    switch (op_tag) {
                        .MATMUL_AB => {
                            try C_grad.bmm_acc_(B.data, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        .MATMUL_AtB => {
                            try B.data.bmm_acc_(C_grad, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        .MATMUL_ABt => {
                            try C_grad.bmm_acc_(B.data, try A.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtBt => {
                            try B.data.bmm_acc_(C_grad, try A.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = true, .beta = 1.0 });
                        },
                        else => unreachable,
                    }
                }

                if (children.get_bwd_upcast(Self, 1)) |_| {
                    const C_grad = C.assume_grad().*;
                    switch (op_tag) {
                        .MATMUL_AB => {
                            try A.data.bmm_acc_(C_grad, try B.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtB => {
                            try A.data.bmm_acc_(C_grad, try B.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_ABt => {
                            try C_grad.bmm_acc_(A.data, try B.ensure_grad(0), C.device, .{ .trans_a = true, .trans_b = false, .beta = 1.0 });
                        },
                        .MATMUL_AtBt => {
                            try C_grad.bmm_acc_(A.data, try B.ensure_grad(0), C.device, .{ .trans_a = false, .trans_b = true, .beta = 1.0 });
                        },
                        else => unreachable,
                    }
                }
            }
        };

        /// Dot product of two tensors. COM.
        pub fn dot(self: *Self, other: *Self) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));

            const DotBwd = struct {
                pub fn backward(c: *Self, children: *Node.Children) !void {
                    scope: {
                        const a = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        c.device.dispatch(opspec.axpy(T){
                            .x = children.get_upcast(Self, 1).get_data(),
                            .y = try a.ensure_grad_data(0),
                            .alpha = @ptrCast(c.assume_grad_data().ptr),
                        });
                    }
                    scope: {
                        const b = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        c.device.dispatch(opspec.axpy(T){
                            .x = children.get_upcast(Self, 0).get_data(),
                            .y = try b.ensure_grad_data(0),
                            .alpha = @ptrCast(c.assume_grad_data().ptr),
                        });
                    }
                }
            };

            return create_dependent(DotBwd, .{
                .data = try self.data.dot(other.data, self.device),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .DOT,
            });
        }

        const MatvecOpts = struct {
            trans_a: bool = false,
        };

        pub fn matvec_(A: *Self, x: *Self, y: *Self, opts: struct {
            trans_a: bool = false,
            beta: T = 0.0,
        }) !void {
            std.debug.assert(A.device.is_compatible(x.device));
            std.debug.assert(A.device.is_compatible(y.device));
            const MatvecBwd = struct {
                _trans_a: bool,
                pub fn backward(_y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    const ta = ctx._trans_a;
                    scope: {
                        const _A = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        const _x = children.get_upcast(Self, 1);
                        _y.device.dispatch(opspec.outer(T){
                            .x = if (ta) _x.get_data() else _y.assume_grad_data(),
                            .y = if (ta) _y.assume_grad_data() else _x.get_data(),
                            .A = try _A.ensure_grad_data(0),
                            .alpha = 1.0,
                        });
                    }
                    scope: {
                        const _x = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        const _A = children.get_upcast(Self, 0);
                        _y.device.dispatch(opspec.matvec(T){
                            .A = _A.get_data(),
                            .x = _y.assume_grad_data(),
                            .y = try _x.ensure_grad_data(0),
                            .m = if (!ta) _A.get_dim(1) else _A.get_dim(0),
                            .n = if (!ta) _A.get_dim(0) else _A.get_dim(1),
                            .trans_a = !ta,
                            .alpha = 1.0,
                            .beta = 1.0,
                        });
                    }
                }
            };

            A.data.matvec_(x.data, &y.data, A.device, .{
                .trans_a = opts.trans_a,
                .alpha = 1.0,
                .beta = opts.beta,
            });

            return prepend_dependent(MatvecBwd, y, .{
                .children = &.{ &A.node, &x.node },
                .callback = .{ ._trans_a = opts.trans_a },
            });
        }

        pub fn matvec(self: *Self, other: *Self, opts: MatvecOpts) !*Self {
            std.debug.assert(self.device.is_compatible(other.device));
            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "trans_a", .value = .{ .bool = opts.trans_a } },
            };

            const MatvecBwd = struct {
                _trans_a: bool,
                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    const ta = ctx._trans_a;
                    scope: {
                        const A = children.get_bwd_upcast(Self, 0) orelse break :scope;
                        const x = children.get_upcast(Self, 1);
                        y.device.dispatch(opspec.outer(T){
                            .x = if (ta) x.get_data() else y.assume_grad_data(),
                            .y = if (ta) y.assume_grad_data() else x.get_data(),
                            .A = try A.ensure_grad_data(0),
                            .alpha = 1.0,
                        });
                    }
                    scope: {
                        const x = children.get_bwd_upcast(Self, 1) orelse break :scope;
                        const A = children.get_upcast(Self, 0);
                        y.device.dispatch(opspec.matvec(T){
                            .A = A.get_data(),
                            .x = y.assume_grad_data(),
                            .y = try x.ensure_grad_data(0),
                            .m = if (!ta) A.get_dim(1) else A.get_dim(0),
                            .n = if (!ta) A.get_dim(0) else A.get_dim(1),
                            .trans_a = !ta,
                            .alpha = 1.0,
                            .beta = 1.0,
                        });
                    }
                }
            };

            return create_dependent(MatvecBwd, .{
                .data = try self.data.matvec(other.data, self.device, .{
                    .trans_a = opts.trans_a,
                    .alpha = 1.0,
                    .beta = 0.0,
                }),
                .children = &.{ &self.node, &other.node },
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{ ._trans_a = opts.trans_a },
                .op = .MATVEC,
                .capture_attributes = &capture_attributes,
            });
        }

        /// Sum of all elements in the tensor. COM.
        pub fn sum(self: *Self) !*Self {
            const SumBwd = struct {
                pub fn backward(y: *Self, children: *Node.Children) !void {
                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    const x_grad = try x.ensure_grad(0);
                    try x_grad._add(y.assume_grad().*, y.device);
                }
            };
            return create_dependent(SumBwd, .{
                .data = try self.data.sum(self.device),
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .op = .SUM,
            });
        }

        /// # ADR
        ///
        /// NOTE: I'm considering several designs right now for the device
        /// layer and how checkpointing should work and what that means for
        /// lifetimes in the autograd system. Additionally, I do not think
        /// we have use cases for differentiable max_along so I'm not going
        /// to implement the backward for now since it will likely end up
        /// needing a refactor.
        pub fn max_along(self: *Self, opts: MaxAlongOptions) !*Self {
            defer if (self.requires_grad()) {
                @panic(
                    \\Do you need differentiable `max_along`? If so, open an
                    \\ issue. If not, disable grad for this op.
                );
            };
            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "dim", .value = .{ .uint = opts.dim } },
                .{ .key = "keep_dims", .value = .{ .bool = opts.keep_dims } },
                .{ .key = "return_indices", .value = .{ .bool = opts.return_indices } },
            };

            // TODO: MaxAlongBwd once I settle on device layer decisions
            const MaxAlongBwd = struct {
                pub fn backward(_: *Self, _: *Node.Children) !void {
                    return error.NotImplemented;
                }
            };
            // const MaxAlongBwd = struct {
            //     indices: []usize,
            //     dim: usize,
            //     src_shape: Shape,
            //     keep_dims: bool,
            //
            //     pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
            //         defer if (ctx.indices.len > 0) y.device.mem_free(ctx.indices);
            //
            //         const input = children.get_bwd_upcast(Self, 0) orelse return;
            //         const grad_output = y.assume_grad_data();
            //         const grad_input = try input.ensure_grad_data(0);
            //
            //         // scatter to max positions
            //         y.device.dispatch(opspec.scatter_add(T){
            //             .src = grad_output,
            //             .offsets = ctx.indices,
            //             .dst = grad_input,
            //         });
            //
            //         // We could be agnostic to checkpointing with a scatter reduce...
            //     }
            // };

            const max_result = try self.data.max_along(self.device, .{
                .dim = opts.dim,
                .keep_dims = opts.keep_dims,
            });

            return create_dependent(MaxAlongBwd, .{
                .data = max_result,
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{},
                .capture_name = "max_along",
                .capture_attributes = &capture_attributes,
            });
        }

        pub fn gather(self: *Self, indices: NDArray(usize), dim: usize) !*Self {
            const capture_attributes = [_]zg.lazy.OpAttribute{
                .{ .key = "dim", .value = .{ .uint = dim } },
                .{ .key = "indices_shape", .value = .{ .usize_list = indices.shape.slice() } },
            };
            var gather_result = try self.data.gather(self.device, .{
                .indices = indices,
                .dim = dim,
                .return_offsets = self.requires_grad(),
            });
            defer if (!self.requires_grad()) gather_result.deinit();

            const GatherBwd = struct {
                offsets: []usize,
                dim: usize,
                src_shape: Shape,

                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    defer y.device.mem_free(ctx.offsets);

                    const input = children.get_bwd_upcast(Self, 0) orelse return;
                    const grad_output = y.assume_grad_data();
                    const grad_input = try input.ensure_grad_data(0);

                    // Accumulate output grads back to input at gathered positions
                    y.device.dispatch(opspec.scatter_add(T){
                        .src = grad_output,
                        .offsets = ctx.offsets,
                        .dst = grad_input,
                    });
                }
            };

            return create_dependent(GatherBwd, .{
                .data = gather_result.values,
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{
                    .offsets = if (self.requires_grad())
                        gather_result.offsets.?
                    else
                        &.{},
                    .dim = dim,
                    .src_shape = Shape.init(self.get_shape()),
                },
                .capture_name = "gather",
                .capture_attributes = &capture_attributes,
            });
        }

        /// Scatter with additive aggregation: dst[indices[i]] += src[i]
        pub fn scatter_add(src: *Self, offsets: []const usize, dst_shape: []const usize) !*Self {
            std.debug.assert(src.data.size() == offsets.len);

            const ScatterAddBwd = struct {
                offsets: []const usize,

                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    defer if (ctx.offsets.len != 0) y.device.mem_free(ctx.offsets);

                    const input_tensor = children.get_bwd_upcast(Self, 0) orelse return;
                    const grad_input = try input_tensor.ensure_grad(0);
                    const grad_output = y.assume_grad();

                    // Gather gradients from scattered positions
                    // grad_output[indices[i]] += grad_input[i]
                    grad_input.scatter_add_(ctx.offsets, grad_output, input_tensor.device);
                }
            };

            var output = try DataType.zeros(dst_shape, src.device);
            src.data.scatter_add_(offsets, &output, src.device);

            const offsets_copy = if (src.requires_grad())
                try src.device.mem_dupe(usize, offsets)
            else
                &.{};

            return create_dependent(ScatterAddBwd, .{
                .data = output,
                .children = &.{&src.node},
                .device = src.device,
                .gb = src.node.gb,
                .callback = .{
                    .offsets = offsets_copy,
                },
                .op = .SCATTER_ADD,
            });
        }

        /// Select complete rows/slices along a dimension using 1D indices
        pub fn index_select(self: *Self, dim: usize, indices: []const usize) !*Self {
            // TODO: lower to device and ndarray layers
            std.debug.assert(dim < self.data.shape.len);

            const IndexSelectBwd = struct {
                dim: usize,
                src_shape: Shape,
                indices_data: []usize,

                pub fn backward(y: *Self, children: *Node.Children, ctx: *@This()) !void {
                    defer y.device.mem_free(ctx.indices_data);

                    const x = children.get_bwd_upcast(Self, 0) orelse return;
                    const x_grad = try x.ensure_grad_data(0);
                    const y_grad = y.assume_grad_data();

                    // Calculate stride
                    const stride = Shape.init(ctx.src_shape.tail(ctx.src_shape.len - ctx.dim - 1)).size();

                    // Scatter gradients back to original positions
                    for (ctx.indices_data, 0..) |src_idx, dst_idx| {
                        const src_start = src_idx * stride;
                        const dst_start = dst_idx * stride;

                        y.device.dispatch(opspec.add(T){
                            .x = y_grad[dst_start .. dst_start + stride],
                            .y = x_grad[src_start .. src_start + stride],
                            .z = x_grad[src_start .. src_start + stride],
                        });
                    }
                }
            };

            const n_indices = indices.len;
            const src_shape = self.data.shape;

            // Calculate output shape
            var out_shape = Shape.init(src_shape.slice());
            out_shape.set(dim, n_indices);

            // Allocate output
            var output = try DataType.empty(out_shape.slice(), self.device);
            errdefer output.deinit(self.device);

            // Stride for dim selecting along
            const stride = Shape.init(src_shape.tail(src_shape.len - dim - 1)).size();

            // Copy data
            const src_data = self.get_data();
            const out_data = output.get_data();
            for (indices, 0..) |src_idx, dst_idx| {
                const src_start = src_idx * stride;
                const dst_start = dst_idx * stride;

                self.device.mem_copy(
                    T,
                    src_data[src_start .. src_start + stride],
                    out_data[dst_start .. dst_start + stride],
                );
            }

            // Store indices for backward pass
            const indices_copy = try self.device.mem_dupe(usize, indices);

            return try create_dependent(IndexSelectBwd, .{
                .data = output,
                .children = &.{&self.node},
                .device = self.device,
                .gb = self.node.gb,
                .callback = .{
                    .dim = dim,
                    .src_shape = src_shape,
                    .indices_data = indices_copy,
                },
                .op = .INDEX_SELECT,
            });
        }

        /// Prints dynamic compuation graph in d2 format with ops as and operands as nodes (non-standard layout)
        /// Prints to stderr using `std.debug.print` for alternatives see `print_to_writer`
        pub fn print_arrows(self: *Self) void {
            var children = self.node.child_iterator() orelse return;
            while (children.next()) |elem| {
                std.debug.print("{?s}<-{?s}", .{ self.get_label(), elem.get_label() });
                const symbol = blk: {
                    const op = self.op orelse break :blk "?";
                    switch (op) {
                        Op.ADD => break :blk "+",
                        Op.SUB => break :blk "-",
                        Op.MUL => break :blk "x",
                        Op.DIV => break :blk "/",
                        Op.SUM => break :blk "++",
                        Op.MATVEC => break :blk "Ax",
                        Op.MATMUL_AB,
                        Op.MATMUL_AtB,
                        Op.MATMUL_ABt,
                        Op.MATMUL_AtBt,
                        => break :blk "AB",
                        else => |o| break :blk @tagName(o),
                    }
                };
                std.debug.print(": {?s}\n", .{symbol});
            }
            var next_children = self.node.child_iterator() orelse return;
            while (next_children.next()) |elem| elem.upcast(Self).print_arrows();
        }

        fn captureStandalone(self: *const Self, capture_name: []const u8) !void {
            if (!zg.lazy.isCapturing()) return;

            try zg.lazy.maybeRecordTensor(.{
                .tensor_key = @intFromPtr(&self.node),
                .op_name = capture_name,
                .dtype_name = @typeName(T),
                .shape = self.get_shape(),
                .device = if (self.device.is_host()) .host else .cuda,
                .requires_grad = self.node.flags.get(.requires_grad),
                .attached = self.node.attached(),
                .acquired = self.node.acquired(),
                .storage = if (self.status == .owned) .owned else .view,
                .label = self.get_label(),
            });
        }

        fn captureTensor(
            self: *const Self,
            capture_name: []const u8,
            children: []const *Node,
            capture_attributes: []const zg.lazy.OpAttribute,
        ) !void {
            if (!zg.lazy.isCapturing()) return;

            var parent_keys = BoundedArray(usize, settings.backward_children_capacity){};
            for (children) |child| {
                const input = child.upcast(Self);
                try input.captureStandalone("external_input");
                try parent_keys.append(@intFromPtr(child));
            }

            try zg.lazy.maybeRecordTensor(.{
                .tensor_key = @intFromPtr(&self.node),
                .parent_keys = parent_keys.slice(),
                .op_name = capture_name,
                .dtype_name = @typeName(T),
                .shape = self.get_shape(),
                .device = if (self.device.is_host()) .host else .cuda,
                .requires_grad = self.node.flags.get(.requires_grad),
                .attached = self.node.attached(),
                .acquired = self.node.acquired(),
                .storage = if (self.status == .owned) .owned else .view,
                .attributes = capture_attributes,
                .label = self.get_label(),
            });
        }

        fn captureMaterialization(self: *const Self, reason: zg.lazy.MaterializationReason) !void {
            if (!zg.lazy.isCapturing()) return;

            try self.captureStandalone("external_input");
            try zg.lazy.maybeRecordMaterialization(@intFromPtr(&self.node), reason);
        }
    };
}

const TestOpts: zg.device.HostDevice.Options = .{
    .small_pool_size = null,
    .large_pool_size = null,
};

test "ndtensor/clamp fw,bw,_clamp,_clamp_grad" {
    const T = f32;
    const Tensor = NDTensor(T);

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    {
        const x = try Tensor.from_slice(cpu.reference(), &.{ -2.0, -0.5, 0.5, 2.0 }, &.{ 2, 2 }, .{
            .requires_grad = true,
            .graph = &graph,
        });
        defer x.deinit();

        const y = try x.clamp(-1.0, 1.0);
        defer y.deinit();

        try y.backward();
        const expected_output: []const f32 = &.{ -1.0, -0.5, 0.5, 1.0 };
        const expected_grad: []const f32 = &.{ 0.0, 1.0, 1.0, 0.0 };

        try std.testing.expectEqualSlices(T, expected_output, y.get_data());
        try std.testing.expectEqualSlices(T, expected_grad, x.assume_grad_data());
    }
}

test "tensor/Graph/sum" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(f32);

    const input = try Tensor.from_slice(cpu.reference(), &.{ 1, 2, 3, 4 }, null, .{
        .requires_grad = true,
        .graph = &graph,
    });
    defer input.deinit();

    const sum_result = try input.sum();
    defer sum_result.deinit();

    try std.testing.expectEqualSlices(f32, &.{10}, sum_result.get_data());

    if (!zg.runtime.grad_enabled) return error.GradNotEnabled;

    try sum_result.backward();

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, input.assume_grad_data());
}

test "tensor/NDTensor index, add, div" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);
    {
        // zig fmt: off
        const t1 = try Tensor.from_slice(device, &.{
             0, 1, 2,
             3, 4, 5,
             6, 7, 8,

             0, 1, 2,
             3, 4, 5,
             6, 7, 8
         }, &.{ 2, 3, 3 }, opts);

        defer t1.deinit();

        const t2 = try Tensor.from_slice(device, &.{ 1, 1, 1 }, null, opts);
        defer t2.deinit();

        const t3 = try Tensor.from_slice(device, &.{
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,

             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
         }, &.{ 2, 3, 3 }, opts);
        defer t3.deinit();

        const t4 = try t1.add(t2);
        defer t4.deinit();

        try t4.backward();

        const t2_grad: [3]f32 = @splat(6);

        try std.testing.expectEqualSlices(f32, t3.get_data(), t4.get_data());
        try std.testing.expectEqualSlices(f32, t2.assume_grad_data(), &t2_grad);
    }
    {
        // zig fmt: off
        const t1 = try Tensor.from_slice(device, &.{
             0, 1, 2,
             3, 4, 5,
             6, 7, 8,

             0, 1, 2,
             3, 4, 5,
             6, 7, 8
         }, &.{ 2, 3, 3 }, opts);
        defer t1.deinit();

        const t2 = try Tensor.from_slice(device, &.{ 1, 1, 1, 1, 1, 1 }, &.{ 2, 1, 3 }, opts);
        defer t2.deinit();

        const t3 = try Tensor.from_slice(device, &.{
             1, 2, 3,
             4, 5, 6,
             7, 8, 9,

             1, 2, 3,
             4, 5, 6,
             7, 8, 9,
         }, &.{ 2, 3, 3 }, opts);
        defer t3.deinit();

        const t4 = try t1.add(t2);
        defer t4.deinit();

        try t4.backward();

        const t2_grad: [6]f32 = @splat(3);

        try std.testing.expectEqualSlices(f32, t3.get_data(), t4.get_data());
        try std.testing.expectEqualSlices(f32, t2.assume_grad_data(), &t2_grad);
    }
}

test "tensor/Graph/addback" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    var t1 = try Tensor.from_slice(device, &.{2.0}, null, opts);
    defer t1.deinit();

    var t2 = try Tensor.from_slice(device, &.{3.0}, null, opts);
    defer t2.deinit();
    // t3 = t1 + t2;
    // dt3/dt1 = 1, dt3/dt2 = 1
    var t3 = try t1.add(t2);
    defer t3.deinit();

    try t3.backward();
    try std.testing.expectEqualDeep(&[_]f32{1.0}, t1.assume_grad_data());
    try std.testing.expectEqualDeep(&[_]f32{1.0}, t2.assume_grad_data());
}

test "tensor/Graph/mulback" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    const t1 = try Tensor.from_slice(device, &.{2}, null, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{3}, null, opts);
    defer t2.deinit();
    // t3 = t1 * t2;
    // dt3/dt1 = t2, dt3/dt2 = t1
    const t3 = try t1.mul(t2);
    defer t3.deinit();

    try t3.backward();

    try std.testing.expectEqualDeep(t2.get_data(), t1.assume_grad_data());
    try std.testing.expectEqualDeep(t1.get_data(), t2.assume_grad_data());
}

test "tensor/Graph/moreback" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    var w = try Tensor.from_slice(device, &.{ 3, 2 }, null, opts);
    defer w.deinit();

    var b = try Tensor.from_slice(device, &.{ 1, 1 }, null, opts);
    defer b.deinit();

    var x = try Tensor.from_slice(device, &.{ 4, 4 }, null, opts);
    defer x.deinit();

    // h = w*x + b
    // dh/dw = x, dh/db = 1
    const temp = try w.mul(x);
    defer temp.deinit();

    const h = try temp.add(b);
    defer h.deinit();

    try h.backward();

    try std.testing.expectEqualSlices(f32, x.get_data(), w.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 1.0 }, b.assume_grad_data());

    // 2 x 1
    const shape2 = &[_]usize{ 2, 1 };
    try w.setup_grad(0);
    try b.setup_grad(0);
    try x.setup_grad(0);
    w._reshape(shape2);
    b._reshape(shape2);
    x._reshape(shape2);
    //// h = w*x + b
    //// dh/dw = x, dh/db = 1
    const temp2 = try w.mul(x);
    defer temp2.deinit();
    const h2 = try temp2.add(b);
    defer h2.deinit();

    try h2.backward();

    try std.testing.expectEqualSlices(f32, x.get_data(), w.assume_grad_data());
    try std.testing.expect(std.mem.allEqual(f32, b.assume_grad_data(), 1));
}

test "tensor/Graph/divback" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const device = cpu.reference();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    var t1 = try Tensor.from_slice(device, &.{ 4, 9 }, null, opts);
    defer t1.deinit();

    var t2 = try Tensor.from_slice(device, &.{ 2, 3 }, null, opts);
    defer t2.deinit();

    var t3 = try t1.div(t2);
    defer t3.deinit();

    try t3.backward();

    const expected_grad_t1 = &[_]f32{ 1.0 / 2.0, 1.0 / 3.0 }; // 1 / b
    const expected_grad_t2 = &[_]f32{ -4.0 / 4.0, -9.0 / 9.0 }; // -a / b^2

    try std.testing.expectEqualSlices(f32, expected_grad_t1, t1.assume_grad_data());
    try std.testing.expectEqualSlices(f32, expected_grad_t2, t2.assume_grad_data());
}

test "tensor/Graph/matmul_backward square" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, opts);
    defer t1.deinit();

    var t2 = try Tensor.from_slice(device, &.{ 1, 0, 0, 1 }, &.{ 2, 2 }, opts);
    defer t2.deinit();

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    try t3.backward();
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 2: Transpose A
    var t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();

    try t3_trans_a.backward();
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 3: Transpose B
    var t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();

    try t3_trans_b.backward();
    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 4: Transpose both A and B
    var t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();

    try t3_trans_ab.backward();
    const expected_grad_t1_trans_ab = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_ab = &[_]T{ 3, 7, 3, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_ab, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_ab, t2.assume_grad_data());
}

test "tensor/Graph/matmul_backward non-square" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const T = f32;
    const Tensor = NDTensor(T);

    // Case 1: No transpose (t1: [2, 2, 3], t2: [2, 3, 2])
    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }, &.{ 2, 2, 3 }, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1 }, &.{ 2, 3, 2 }, opts);
    defer t2.deinit();

    // Case 1: No transpose
    {
        const t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
        defer t3.deinit();

        try t3.backward();
        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.assume_grad_data());
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.assume_grad_data());
        try t1.setup_grad(0);
        try t2.setup_grad(0);
    }

    // Case 2: Transpose A (t1: [2, 3, 2], t2: [2, 3, 2])
    {
        const t1_case2 = try Tensor.from_slice(device, &.{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }, &.{ 2, 3, 2 }, opts);
        defer t1_case2.deinit();

        var t3 = try t1_case2.bmm(t2, .{ .trans_a = true, .trans_b = false });
        defer t3.deinit();

        try t3.backward();
        const expected_grad_t1 = &[_]T{ 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2 };
        const expected_grad_t2 = &[_]T{ 5, 5, 7, 7, 9, 9, 17, 17, 19, 19, 21, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1_case2.assume_grad_data());
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2.assume_grad_data());
        try t2.setup_grad(0);
    }

    // Case 3: Transpose B (t1: [2, 2, 3], t2: [2, 2, 3])
    {
        var t2_case3 = try Tensor.from_slice(device, &.{ 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 }, &.{ 2, 2, 3 }, opts);
        defer t2_case3.deinit();

        var t3 = try t1.bmm(t2_case3, .{ .trans_a = false, .trans_b = true });
        defer t3.deinit();

        try t3.backward();
        const expected_grad_t1 = &[_]T{ 1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2 };
        const expected_grad_t2 = &[_]T{ 5, 7, 9, 5, 7, 9, 17, 19, 21, 17, 19, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1.assume_grad_data());
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2_case3.assume_grad_data());
        try t1.setup_grad(0);
    }

    // Case 4: Transpose both A and B (t1: [2, 3, 2], t2: [2, 2, 3])
    {
        const t1_case4 = try Tensor.from_slice(device, &.{ 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }, &.{ 2, 3, 2 }, opts);
        defer t1_case4.deinit();

        const t2_case4 = try Tensor.from_slice(device, &.{ 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1 }, &.{ 2, 2, 3 }, opts);
        defer t2_case4.deinit();

        var t3 = try t1_case4.bmm(t2_case4, .{ .trans_a = true, .trans_b = true });
        defer t3.deinit();

        try t3.backward();
        const expected_grad_t1 = &[_]T{ 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2 };
        const expected_grad_t2 = &[_]T{ 5, 7, 9, 5, 7, 9, 17, 19, 21, 17, 19, 21 };
        try std.testing.expectEqualSlices(T, expected_grad_t1, t1_case4.assume_grad_data());
        try std.testing.expectEqualSlices(T, expected_grad_t2, t2_case4.assume_grad_data());
    }
}

test "tensor/Graph/matmul_backward nested broadcast batches" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const T = f32;
    const Tensor = NDTensor(T);

    var t1 = try Tensor.from_slice(device, &.{
        1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24,
    }, &.{ 2, 2, 2, 3 }, opts);
    defer t1.deinit();

    var t2 = try Tensor.from_slice(device, &.{
        1,  2,  3,  4,  5,  6,
        7,  8,  9,  10, 11, 12,
    }, &.{ 2, 1, 3, 2 }, opts);
    defer t2.deinit();

    var t3 = try t1.bmm(t2, .{});
    defer t3.deinit();

    try t3.backward();

    const expected_out = &[_]T{
        22, 28, 49, 64, 76, 100, 103, 136,
        382, 424, 463, 514, 544, 604, 625, 694,
    };
    const expected_grad_t1 = &[_]T{
        3,  7,  11, 3,  7,  11,
        3,  7,  11, 3,  7,  11,
        15, 19, 23, 15, 19, 23,
        15, 19, 23, 15, 19, 23,
    };
    const expected_grad_t2 = &[_]T{
        22, 22, 26, 26, 30, 30,
        70, 70, 74, 74, 78, 78,
    };

    try std.testing.expectEqualSlices(T, expected_out, t3.get_data());
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.assume_grad_data());
}

test "tensor/Graph/matmul_backward" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const T = f32;
    const Tensor = NDTensor(T);
    const shape = &[_]usize{ 2, 2 };

    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, shape, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 0, 0, 1 }, shape, opts);
    defer t2.deinit();

    // Case 1: No transpose
    var t3 = try t1.bmm(t2, .{ .trans_a = false, .trans_b = false });
    defer t3.deinit();

    try t3.backward();
    const expected_grad_t1 = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2 = &[_]T{ 4, 4, 6, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 2: Transpose A
    const t3_trans_a = try t1.bmm(t2, .{ .trans_a = true, .trans_b = false });
    defer t3_trans_a.deinit();

    try t3_trans_a.backward();
    const expected_grad_t1_trans_a = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_a = &[_]T{ 3, 3, 7, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_a, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_a, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 3: Transpose B
    const t3_trans_b = try t1.bmm(t2, .{ .trans_a = false, .trans_b = true });
    defer t3_trans_b.deinit();

    try t3_trans_b.backward();
    const expected_grad_t1_trans_b = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_b = &[_]T{ 4, 6, 4, 6 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_b, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_b, t2.assume_grad_data());
    try t1.setup_grad(0);
    try t2.setup_grad(0);

    // Case 4: Transpose both A and B
    const t3_trans_ab = try t1.bmm(t2, .{ .trans_a = true, .trans_b = true });
    defer t3_trans_ab.deinit();

    try t3_trans_ab.backward();
    const expected_grad_t1_trans_ab = &[_]T{ 1, 1, 1, 1 };
    const expected_grad_t2_trans_ab = &[_]T{ 3, 7, 3, 7 };
    try std.testing.expectEqualSlices(T, expected_grad_t1_trans_ab, t1.assume_grad_data());
    try std.testing.expectEqualSlices(T, expected_grad_t2_trans_ab, t2.assume_grad_data());
}

test "tensor/Graph/matvec_backward" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    // [1, 2] [1]
    // [3, 4] [1]
    // grad = [1, 1]'
    // dl/dA = grad * [1, 1] = [[2, 2], [2, 2]]
    // dl/dx = A' * grad = [4, 6]'
    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{2, 2}, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 1, 1 }, &.{2}, opts);
    defer t2.deinit();

    const t3 = try t1.matvec(t2, .{});
    defer t3.deinit();

    try t3.backward();

    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, t1.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{4, 6}, t2.assume_grad_data());
}

test "tensor/Graph/dot_backward" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    const t1 = try Tensor.from_slice(device, &.{ 1, 2, 3 }, null, opts);
    defer t1.deinit();

    const t2 = try Tensor.from_slice(device, &.{ 4, 5, 6 }, null, opts);
    defer t2.deinit();

    var t3 = try t1.dot(t2);
    defer t3.deinit();

    try t3.backward();

    try std.testing.expectEqualSlices(f32, &.{4, 5, 6}, t1.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{1, 2, 3}, t2.assume_grad_data());
}


test "tensor/inplace_add" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const u = try NDTensor(f32).ones(device, &.{ 2, 2 }, opts);
    defer u.deinit();

    const v = try NDTensor(f32).ones(device, &.{ 2, 2 }, opts);
    defer v.deinit();

    const x = try u.mul(v);
    defer x.deinit();

    const a = try NDTensor(f32).ones(device, &.{ 2, 2 }, opts);
    defer a.deinit();

    const b = try NDTensor(f32).ones(device, &.{ 2, 2 }, opts);
    defer b.deinit();

    const c = try NDTensor(f32).ones(device, &.{ 2, 2 }, opts);
    defer c.deinit();

    // x now carries 4 contexts for (a), (b), (c), (u, v)
    try a.add_(x);
    try b.add_(x);
    try c.add_(x);

    try x.setup_grad(2.0);

    try x.backward();
    try std.testing.expectEqualSlices(f32, &.{ 4, 4, 4, 4 }, x.get_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, a.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, b.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, c.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, u.assume_grad_data());
    try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2 }, v.assume_grad_data());

    // check the children...
    var children = x.node.child_iterator() orelse unreachable;
    try std.testing.expectEqual(children.next().?, &c.node);
    try std.testing.expectEqual(children.next().?, &b.node);
    try std.testing.expectEqual(children.next().?, &a.node);
    try std.testing.expectEqual(children.next().?, &u.node);
    try std.testing.expectEqual(children.next().?, &v.node);
    try std.testing.expectEqual(children.next(), null);
}

test "tensor/Graph/subset" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(f32);

    const t1 = try Tensor.from_slice(device, &.{ 1, 1, 1, 1, 1, 2, 2, 2, 2, 2 }, &.{ 2, 5 }, .{
        .requires_grad = true,
        .graph = &graph,
    });
    defer t1.deinit();

    {
        const t2 = try t1.subset(&.{ 1 }, .view);
        defer t2.deinit();

        try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2, 2 }, t2.get_data());

        try t2.backward();

        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, t1.assume_grad_data());
    }

    try t1.setup_grad(0);

    {
        const t2 = try  t1.subset(&.{ 1 }, .owned);
        defer t2.deinit();

        try std.testing.expectEqualSlices(f32, &.{ 2, 2, 2, 2, 2 }, t2.get_data());

        try t2.backward();

        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, t1.assume_grad_data());
    }

}

test "tensor/Graph/getter-setter" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(f32);

    const t1 = try Tensor.from_slice(device, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, &.{ 2, 5 }, .{ .graph = &graph });
    defer t1.deinit();

    {
        const t2 = try Tensor.from_slice(device, &.{ 1, 1, 1, 1, 1 }, null, .{ .graph = &graph });
        defer t2.deinit();

        t1.set_offset(5, t2);

        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 }, t1.get_data());
    }
    {
        const t2 = try  Tensor.empty(device, &.{ 5 }, .{ .graph = &graph });
        defer t2.deinit();

        t1.get_offset(5, t2);

        try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 1 }, t2.get_data());
    }

}

test "tensor/to_host_owned duplicates tensor contents" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(f32);
    const tensor = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{ .graph = &graph });
    defer tensor.deinit();

    const host = try tensor.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(host);

    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, host);
}

test "tensor/lazy session captures eager ops and materialization boundaries" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const Tensor = NDTensor(f32);

    const lhs = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .requires_grad = true,
        .graph = &graph,
        .label = "lhs",
    });
    defer lhs.deinit();

    const rhs = try Tensor.from_slice(device, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }, .{
        .requires_grad = true,
        .graph = &graph,
        .label = "rhs",
    });
    defer rhs.deinit();

    const sum = try lhs.add(rhs);
    defer sum.deinit();

    const view = try sum.alias();
    defer view.deinit();

    _ = try view.realize();

    const host = try view.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(host);

    try std.testing.expectEqualSlices(f32, &.{ 6, 8, 10, 12 }, host);
    try std.testing.expectEqual(@as(usize, 4), session.tensors().len);
    try std.testing.expectEqual(@as(usize, 2), session.materializationEvents().len);

    const records = session.tensors();
    try std.testing.expectEqualStrings("source", records[0].op_name);
    try std.testing.expectEqualStrings("source", records[1].op_name);
    try std.testing.expectEqualStrings("ADD", records[2].op_name);
    try std.testing.expectEqualStrings("alias", records[3].op_name);
    try std.testing.expectEqual(zg.lazy.StorageKind.view, records[3].storage);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2 }, records[2].parent_ids);
    try std.testing.expectEqualSlices(u32, &.{3}, records[3].parent_ids);

    const materializations = session.materializationEvents();
    try std.testing.expectEqual(zg.lazy.MaterializationReason.explicit, materializations[0].reason);
    try std.testing.expectEqual(zg.lazy.MaterializationReason.host_read, materializations[1].reason);
    try std.testing.expectEqual(@as(u32, 4), materializations[0].tensor_id);

    var summary = std.ArrayList(u8){};
    defer summary.deinit(std.testing.allocator);
    try session.writeSummary(summary.writer(std.testing.allocator));
    try std.testing.expect(std.mem.indexOf(u8, summary.items, "op=ADD") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.items, "storage=view") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.items, "materialize tensor=#4 reason=explicit") != null);

    var d2 = std.ArrayList(u8){};
    defer d2.deinit(std.testing.allocator);
    try session.writeD2(d2.writer(std.testing.allocator));
    try std.testing.expect(std.mem.indexOf(u8, d2.items, "t1 -> t3") != null);
    try std.testing.expect(std.mem.indexOf(u8, d2.items, "t3 -> t4") != null);
    try std.testing.expect(std.mem.indexOf(u8, d2.items, "t4 -> m1") != null);
}

test "tensor/lazy session treats preexisting tensors as external inputs" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const Tensor = NDTensor(f32);
    const input = try Tensor.from_slice(device, &.{ 1, 4, 9, 16 }, &.{ 2, 2 }, .{
        .requires_grad = true,
        .graph = &graph,
        .label = "preexisting",
    });
    defer input.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    try std.testing.expectEqual(@as(usize, 0), session.tensors().len);

    var capture = try session.begin();
    defer capture.end();

    const output = try input.sqrt();
    defer output.deinit();

    const records = session.tensors();
    try std.testing.expectEqual(@as(usize, 2), records.len);
    try std.testing.expectEqualStrings("external_input", records[0].op_name);
    try std.testing.expectEqualStrings("sqrt", records[1].op_name);
    try std.testing.expectEqualSlices(u32, &.{1}, records[1].parent_ids);
    try std.testing.expectEqualStrings("preexisting", records[0].label.?);
}

test "tensor/lazy session records op attributes and json dump" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const Tensor = NDTensor(f32);

    const input = try Tensor.from_slice(device, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 }, .{
        .graph = &graph,
        .label = "input",
    });
    defer input.deinit();

    const reshaped = try input.reshape(&.{ 3, 2 });
    defer reshaped.deinit();

    const transposed = try reshaped.transpose();
    defer transposed.deinit();

    const rhs = try Tensor.from_slice(device, &.{ 1, 0, 0, 1, 1, 1 }, &.{ 2, 3 }, .{
        .graph = &graph,
        .label = "rhs",
    });
    defer rhs.deinit();

    const matmul = try transposed.bmm(rhs, .{
        .trans_b = true,
        .alpha = 0.5,
        .beta = 0.25,
    });
    defer matmul.deinit();

    const subset = try transposed.subset(&.{1}, .view);
    defer subset.deinit();

    const maxed = try transposed.max_along(.{
        .dim = 1,
        .keep_dims = true,
    });
    defer maxed.deinit();

    const softmaxed = try zg.loss.softmax(f32, transposed, 1, device);
    defer softmaxed.deinit();

    const records = session.tensors();

    const reshape_record = findLazyRecord(records, "RESHAPE") orelse return error.MissingLazyReshapeRecord;
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, findLazyUsizeListAttribute(reshape_record.attributes, "new_shape") orelse return error.MissingLazyReshapeAttribute);

    const transpose_record = findLazyRecord(records, "TRANSPOSE") orelse return error.MissingLazyTransposeRecord;
    try std.testing.expectEqualSlices(usize, &.{ 1, 0 }, findLazyUsizeListAttribute(transpose_record.attributes, "permutation") orelse return error.MissingLazyTransposeAttribute);

    const matmul_record = findLazyRecord(records, "MATMUL_ABt") orelse return error.MissingLazyMatmulRecord;
    try std.testing.expectEqual(false, findLazyBoolAttribute(matmul_record.attributes, "trans_a") orelse return error.MissingLazyMatmulTransAAttribute);
    try std.testing.expectEqual(true, findLazyBoolAttribute(matmul_record.attributes, "trans_b") orelse return error.MissingLazyMatmulTransBAttribute);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), findLazyFloatAttribute(matmul_record.attributes, "alpha") orelse return error.MissingLazyMatmulAlphaAttribute, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), findLazyFloatAttribute(matmul_record.attributes, "beta") orelse return error.MissingLazyMatmulBetaAttribute, 1e-9);

    const subset_record = findLazyRecord(records, "subset") orelse return error.MissingLazySubsetRecord;
    try std.testing.expectEqualSlices(i64, &.{1}, findLazyIntListAttribute(subset_record.attributes, "steps") orelse return error.MissingLazySubsetStepsAttribute);

    const max_record = findLazyRecord(records, "max_along") orelse return error.MissingLazyMaxAlongRecord;
    try std.testing.expectEqual(@as(u64, 1), findLazyUintAttribute(max_record.attributes, "dim") orelse return error.MissingLazyMaxAlongDimAttribute);
    try std.testing.expectEqual(true, findLazyBoolAttribute(max_record.attributes, "keep_dims") orelse return error.MissingLazyMaxAlongKeepDimsAttribute);
    try std.testing.expectEqual(false, findLazyBoolAttribute(max_record.attributes, "return_indices") orelse return error.MissingLazyMaxAlongReturnIndicesAttribute);

    const softmax_record = findLazyRecord(records, "softmax") orelse return error.MissingLazySoftmaxRecord;
    try std.testing.expectEqual(@as(u64, 1), findLazyUintAttribute(softmax_record.attributes, "dim") orelse return error.MissingLazySoftmaxDimAttribute);

    var summary = std.ArrayList(u8){};
    defer summary.deinit(std.testing.allocator);
    try session.writeSummary(summary.writer(std.testing.allocator));
    try std.testing.expect(std.mem.indexOf(u8, summary.items, "attrs={new_shape=[3,2]}") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.items, "attrs={trans_a=false,trans_b=true,alpha=0.5,beta=0.25}") != null);

    var json_writer = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer json_writer.deinit();
    try session.writeJson(&json_writer.writer);
    const json_bytes = json_writer.written();
    try std.testing.expect(std.mem.indexOf(u8, json_bytes, "\"attributes\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json_bytes, "\"key\":\"new_shape\"") != null);

    var json_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer json_arena.deinit();
    const parsed = try std.json.parseFromSliceLeaky(zg.lazy.SessionDump, json_arena.allocator(), json_bytes, .{
        .ignore_unknown_fields = false,
    });
    const parsed_subset = findLazyRecord(parsed.tensors, "subset") orelse return error.MissingParsedLazySubsetRecord;
    try std.testing.expectEqualSlices(i64, &.{1}, findLazyIntListAttribute(parsed_subset.attributes, "steps") orelse return error.MissingParsedLazySubsetStepsAttribute);
}

fn findLazyRecord(records: []const zg.lazy.TensorRecord, op_name: []const u8) ?*const zg.lazy.TensorRecord {
    for (records) |*record| {
        if (std.mem.eql(u8, record.op_name, op_name)) return record;
    }
    return null;
}

fn findLazyAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?zg.lazy.AttributeValue {
    for (attributes) |attribute| {
        if (std.mem.eql(u8, attribute.key, key)) return attribute.value;
    }
    return null;
}

fn findLazyBoolAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?bool {
    const value = findLazyAttribute(attributes, key) orelse return null;
    return switch (value) {
        .bool => |item| item,
        else => null,
    };
}

fn findLazyUintAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?u64 {
    const value = findLazyAttribute(attributes, key) orelse return null;
    return switch (value) {
        .uint => |item| item,
        else => null,
    };
}

fn findLazyFloatAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?f64 {
    const value = findLazyAttribute(attributes, key) orelse return null;
    return switch (value) {
        .float => |item| item,
        else => null,
    };
}

fn findLazyUsizeListAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?[]const usize {
    const value = findLazyAttribute(attributes, key) orelse return null;
    return switch (value) {
        .usize_list => |item| item,
        else => null,
    };
}

fn findLazyIntListAttribute(attributes: []const zg.lazy.OpAttribute, key: []const u8) ?[]const i64 {
    const value = findLazyAttribute(attributes, key) orelse return null;
    return switch (value) {
        .int_list => |item| item,
        else => null,
    };
}

test "tensor/deferred lazy session produces same results as eager" {
    const T = f32;
    const Tensor = NDTensor(T);

    // Eager reference run
    var eager_cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer eager_cpu.deinit();
    const eager_device = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const eager_a = try Tensor.from_slice(eager_device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .graph = &eager_graph,
    });
    defer eager_a.deinit();

    const eager_b = try Tensor.from_slice(eager_device, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }, .{
        .graph = &eager_graph,
    });
    defer eager_b.deinit();

    const eager_sum = try eager_a.add(eager_b);
    defer eager_sum.deinit();

    const eager_prod = try eager_sum.mul(eager_a);
    defer eager_prod.deinit();

    const eager_result = try eager_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // Deferred run
    var def_cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer def_cpu.deinit();
    const def_device = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const def_a = try Tensor.from_slice(def_device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .graph = &def_graph,
    });
    defer def_a.deinit();

    const def_b = try Tensor.from_slice(def_device, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }, .{
        .graph = &def_graph,
    });
    defer def_b.deinit();

    const def_sum = try def_a.add(def_b);
    defer def_sum.deinit();

    const def_prod = try def_sum.mul(def_a);
    defer def_prod.deinit();

    // Before realize, thunks should be pending
    try std.testing.expect(session.pendingThunkCount() > 0);

    // realize() flushes deferred work
    _ = try def_prod.realize();

    // After realize, thunks should be drained
    try std.testing.expectEqual(@as(usize, 0), session.pendingThunkCount());

    const def_result = try def_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(def_result);

    try std.testing.expectEqualSlices(T, eager_result, def_result);
}

test "tensor/deferred lazy session auto-realizes on host read" {
    const T = f32;
    const Tensor = NDTensor(T);

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 10, 20, 30, 40 }, null, .{
        .graph = &graph,
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, null, .{
        .graph = &graph,
    });
    defer b.deinit();

    const result = try a.sub(b);
    defer result.deinit();

    // Thunks are pending
    try std.testing.expect(session.pendingThunkCount() > 0);

    // copy_to_host auto-flushes deferred thunks
    const host = try result.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(host);

    try std.testing.expectEqualSlices(T, &.{ 9, 18, 27, 36 }, host);
    try std.testing.expectEqual(@as(usize, 0), session.pendingThunkCount());
}

test "tensor/deferred lazy session multi-op chain with transpose and matmul" {
    const T = f32;
    const Tensor = NDTensor(T);

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    // 2x3 matrix
    const a = try Tensor.from_slice(device, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 }, .{
        .graph = &graph,
    });
    defer a.deinit();

    // transpose to 3x2
    const at = try a.transpose();
    defer at.deinit();

    // 2x3 @ 3x2 = 2x2
    const prod = try a.bmm(at, .{});
    defer prod.deinit();

    // Should have queued thunks
    try std.testing.expect(session.pendingThunkCount() > 0);

    const host = try prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(host);

    // A @ A^T = [[1*1+2*2+3*3, 1*4+2*5+3*6], [4*1+5*2+6*3, 4*4+5*5+6*6]]
    //         = [[14, 32], [32, 77]]
    try std.testing.expectEqualSlices(T, &.{ 14, 32, 32, 77 }, host);
}

test "tensor/observe mode unchanged by deferred infrastructure" {
    const T = f32;
    const Tensor = NDTensor(T);

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    // mode defaults to .observe — no deferred execution

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, null, .{
        .graph = &graph,
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 5, 6, 7, 8 }, null, .{
        .graph = &graph,
    });
    defer b.deinit();

    const result = try a.add(b);
    defer result.deinit();

    // In observe mode, no thunks are queued
    try std.testing.expectEqual(@as(usize, 0), session.pendingThunkCount());

    // Data is available immediately (eager execution)
    try std.testing.expectEqualSlices(T, &.{ 6, 8, 10, 12 }, result.get_data());
}

test "tensor/deferred session records capture metadata alongside thunks" {
    const T = f32;
    const Tensor = NDTensor(T);

    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .graph = &graph,
        .label = "a",
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }, .{
        .graph = &graph,
        .label = "b",
    });
    defer b.deinit();

    const sum = try a.add(b);
    defer sum.deinit();

    _ = try sum.realize();

    // Session should have captured tensor records even in deferred mode
    const records = session.tensors();
    try std.testing.expect(records.len >= 3);
    try std.testing.expectEqualStrings("source", records[0].op_name);
    try std.testing.expectEqualStrings("source", records[1].op_name);
    try std.testing.expectEqualStrings("ADD", records[2].op_name);

    // Materialization event should be recorded
    const materializations = session.materializationEvents();
    try std.testing.expect(materializations.len >= 1);
    try std.testing.expectEqual(zg.lazy.MaterializationReason.explicit, materializations[0].reason);

    // And the result should be correct
    const host = try sum.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(host);
    try std.testing.expectEqualSlices(T, &.{ 6, 8, 10, 12 }, host);
}

test "tensor/pow" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    const x = try Tensor.from_slice(device, &.{2, 3, 0, 1}, null, opts);
    defer x.deinit();

    const out = try x.pow(2);
    defer out.deinit();
    try out.backward();

    try std.testing.expectEqualDeep(&[_]f32{4, 9, 0, 1}, out.get_data());
    try std.testing.expectEqualDeep(&[_]f32{4, 6, 0, 2}, x.assume_grad_data());
}

test "tensor/sqrt" {
    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();

    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    const opts: TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const Tensor = NDTensor(f32);

    const x = try Tensor.from_slice(device, &.{4, 9, 16, 1}, null, opts);
    defer x.deinit();

    const out = try x.sqrt();
    defer out.deinit();

    try out.backward();

    try std.testing.expectEqualSlices(f32, &[_]f32{2, 3, 4, 1}, out.get_data());
    // d/dx[sqrt(x)] = 1/(2*sqrt(x))
    try std.testing.expectEqualSlices(f32, &[_]f32{0.25, 1.0/6.0, 0.125, 0.5}, x.assume_grad_data());
}

// TODO: Fix memory freeing conundrum with gather() then dont use an arena here.;;
//test "tensor/gather" {;;
//    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//    defer arena.deinit();
//    const allocator = arena.allocator();
//
//    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
//    defer cpu.deinit();
//    const device = cpu.reference();
//
//    const T = f32;
//    const Tensor = NDTensor(T);
//
//    // case 1: basic gather;;
//    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//    const input_shape = [_]usize{ 3, 3 };
//    var input = try Tensor.from_slice(&input_data, &input_shape, true, device);
//    defer input.deinit();
//
//    const index_data = [_]usize{ 0, 1, 1, 2, 0, 2 };
//    const index_shape = [_]usize{ 3, 2 };
//    var index = try NDTensor(usize).init(&index_data, &index_shape, false, device);
//    defer index.deinit();
//
//    var output = try input.gather(device, .{ .indices = index, .dim = 1 });;;
//    defer output.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 1, 2, 5, 6, 7, 9 }, output.data.data);
//    try std.testing.expectEqualSlices(usize, &index_shape, output.data.shape.slice());
//
//    // case 2: grad check
//    var gm = Graph(Tensor).init(device.allocator, .{});
//    defer gm.deinit();
//
//    output.grad.?.fill(1.0, device);
//    try gm.backward(output);
//
//    const expected_grad = [_]T{ 1, 1, 0, 0, 1, 1, 1, 0, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);
//
//    // case 3: out of bounds
//    //try index.set(&.{ 0, 0 }, 3);
//    //try std.testing.expectError(error.IndexOutOfBounds, input.gather(device, .{ .indices = index, .dim = 1 }));;;
//}

// TODO: Fix memory freeing conundrum with max_over_dim() then dont use an arena here.
//test "tensor/max_over_dim" {
//    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
//    defer arena.deinit();
//    const allocator = arena.allocator();
//
//    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
//    defer cpu.deinit();
//    const device = cpu.reference();
//
//    const T = f32;
//    const Tensor = NDTensor(T);
//
//    // case 1: basic max over dim operation
//    const input_data = [_]T{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
//    const input_shape = [_]usize{ 3, 3 };
//    var input = try Tensor.from_slice(&input_data, &input_shape, true, device);
//    defer input.deinit();
//
//    var output = try input.max_over_dim(device, .{ .dim = 1 });
//    defer output.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 3, 6, 9 }, output.data.data);
//    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output.data.shape.shape);
//
//    // case 2: gradient check
//    var gm = Graph(Tensor).init(device.allocator, .{});
//    defer gm.deinit();
//
//    output.grad.?.fill(1.0, device);
//    try gm.backward(output);
//
//    const expected_grad = [_]T{ 0, 0, 1, 0, 0, 1, 0, 0, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad, input.grad.?.data);
//
//    // case 3: max over different dimension
//    var output2 = try input.max_over_dim(device, .{ .dim = 0 });
//    defer output2.deinit();
//
//    try std.testing.expectEqualSlices(T, &[_]T{ 7, 8, 9 }, output2.data.data);
//    try std.testing.expectEqualSlices(usize, &[_]usize{3}, output2.data.shape.shape);
//
//    // reset grads
//    input.grad.?.fill(0, device);
//    output2.grad.?.fill(1.0, device);
//    try gm.backward(output2);
//
//    const expected_grad2 = [_]T{ 0, 0, 0, 0, 0, 0, 1, 1, 1 };
//    try std.testing.expectEqualSlices(T, &expected_grad2, input.grad.?.data);
//
//    // case 4: invalid dimension
//    try std.testing.expectError(error.DimOutOfBounds, input.max_over_dim(device, .{ .dim = 2 }));
//}
