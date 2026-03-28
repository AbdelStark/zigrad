const std = @import("std");
const result = @import("result.zig");

pub const Suite = enum {
    primitive,
    blas,
    autograd,
    memory,
    model_train,
    model_infer,

    pub fn asString(self: Suite) []const u8 {
        return switch (self) {
            .primitive => "primitive",
            .blas => "blas",
            .autograd => "autograd",
            .memory => "memory",
            .model_train => "model-train",
            .model_infer => "model-infer",
        };
    }
};

pub const Kind = enum {
    primitive_add,
    primitive_matmul,
    blas_dot,
    blas_matvec,
    blas_conv2d_im2col,
    autograd_dot_backward,
    autograd_matvec_backward,
    memory_tensor_cache_cycle,
    memory_mnist_train_step,
    mnist_mlp_train,
    mnist_mlp_infer,
    dqn_cartpole_train,
    dqn_cartpole_infer,
    gcn_train,
    gcn_infer,

    pub fn asString(self: Kind) []const u8 {
        return switch (self) {
            .primitive_add => "primitive_add",
            .primitive_matmul => "primitive_matmul",
            .blas_dot => "blas_dot",
            .blas_matvec => "blas_matvec",
            .blas_conv2d_im2col => "blas_conv2d_im2col",
            .autograd_dot_backward => "autograd_dot_backward",
            .autograd_matvec_backward => "autograd_matvec_backward",
            .memory_tensor_cache_cycle => "memory_tensor_cache_cycle",
            .memory_mnist_train_step => "memory_mnist_train_step",
            .mnist_mlp_train => "mnist_mlp_train",
            .mnist_mlp_infer => "mnist_mlp_infer",
            .dqn_cartpole_train => "dqn_cartpole_train",
            .dqn_cartpole_infer => "dqn_cartpole_infer",
            .gcn_train => "gcn_train",
            .gcn_infer => "gcn_infer",
        };
    }
};

pub const DType = enum {
    f32,

    pub fn asString(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
        };
    }
};

pub const DeviceKind = enum {
    host,
    cuda,

    pub fn asString(self: DeviceKind) []const u8 {
        return switch (self) {
            .host => "host",
            .cuda => "cuda",
        };
    }
};

pub const DeviceRequest = struct {
    kind: DeviceKind = .host,
    cuda_device_index: u32 = 0,

    pub fn parse(value: ?[]const u8) !DeviceRequest {
        const trimmed = std.mem.trim(u8, value orelse return .{}, " \t\r\n");
        if (trimmed.len == 0) return .{};

        if (std.ascii.eqlIgnoreCase(trimmed, "host") or std.ascii.eqlIgnoreCase(trimmed, "cpu")) {
            return .{ .kind = .host };
        }
        if (std.ascii.eqlIgnoreCase(trimmed, "cuda")) {
            return .{ .kind = .cuda, .cuda_device_index = 0 };
        }
        if (trimmed.len > 5 and std.ascii.eqlIgnoreCase(trimmed[0..5], "cuda:")) {
            return .{
                .kind = .cuda,
                .cuda_device_index = std.fmt.parseInt(u32, trimmed[5..], 10) catch return error.InvalidBenchmarkDevice,
            };
        }

        return error.InvalidBenchmarkDevice;
    }
};

pub const Spec = struct {
    id: []const u8,
    suite: Suite,
    kind: Kind,
    device: DeviceRequest = .{},
    dtype: DType,
    warmup_iterations: u32 = 5,
    measured_iterations: u32 = 20,
    batch_size: ?usize = null,
    thread_count: ?u32 = null,
    seed: u64 = 81761,
    lhs_shape: ?[]const usize = null,
    rhs_shape: ?[]const usize = null,
    input_shape: ?[]const usize = null,
    label_shape: ?[]const usize = null,
    output_shape: ?[]const usize = null,
    stride: usize = 1,
    padding: usize = 0,
    dilation: usize = 1,
    provenance: result.BenchmarkProvenance,
    notes: ?[]const u8 = null,
    pytorch_runner: ?[]const u8 = null,
    path: []const u8,
};

const RawBenchmarkProvenance = struct {
    data_source: ?[]const u8 = null,
    preprocessing: ?[]const []const u8 = null,
};

const RawSpec = struct {
    id: []const u8,
    suite: []const u8,
    kind: []const u8,
    device: ?[]const u8 = null,
    dtype: []const u8 = "f32",
    warmup_iterations: u32 = 5,
    measured_iterations: u32 = 20,
    batch_size: ?usize = null,
    thread_count: ?u32 = null,
    seed: u64 = 81761,
    lhs_shape: ?[]usize = null,
    rhs_shape: ?[]usize = null,
    input_shape: ?[]usize = null,
    label_shape: ?[]usize = null,
    output_shape: ?[]usize = null,
    stride: usize = 1,
    padding: usize = 0,
    dilation: usize = 1,
    provenance: ?RawBenchmarkProvenance = null,
    notes: ?[]const u8 = null,
    pytorch_runner: ?[]const u8 = null,
};

pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !Spec {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const bytes = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    const raw = try std.json.parseFromSliceLeaky(RawSpec, allocator, bytes, .{});
    return try validate(path, raw);
}

fn validate(path: []const u8, raw: RawSpec) !Spec {
    const suite = try parseSuite(raw.suite);
    const kind = try parseKind(raw.kind);
    const device = try DeviceRequest.parse(raw.device);
    const dtype = try parseDType(raw.dtype);
    const provenance = try validateProvenance(raw.provenance orelse return error.MissingBenchmarkProvenance);

    if (raw.measured_iterations == 0) return error.InvalidBenchmarkIterations;

    switch (kind) {
        .primitive_add, .primitive_matmul => {
            if (raw.lhs_shape == null or raw.rhs_shape == null) {
                return error.MissingPrimitiveShape;
            }
        },
        .blas_dot, .autograd_dot_backward => {
            try requireDotShapes(raw);
        },
        .blas_matvec, .autograd_matvec_backward => {
            try requireMatvecShapes(raw);
        },
        .blas_conv2d_im2col => {
            try requireConv2dShapes(raw);
        },
        .memory_tensor_cache_cycle => {
            if (raw.lhs_shape == null) return error.MissingPrimitiveShape;
            if (raw.batch_size == null or raw.batch_size.? == 0) return error.MissingBatchSize;
        },
        .memory_mnist_train_step => {
            try requireBatchedModelShapes(raw, true);
        },
        .mnist_mlp_train => {
            try requireBatchedModelShapes(raw, true);
        },
        .mnist_mlp_infer => {
            try requireBatchedModelShapes(raw, false);
        },
        .dqn_cartpole_train => {
            try requireBatchedModelShapes(raw, false);
        },
        .dqn_cartpole_infer => {
            try requireBatchedModelShapes(raw, false);
        },
        .gcn_train => {
            if (raw.input_shape == null or raw.label_shape == null) {
                return error.MissingModelShape;
            }
            if (raw.input_shape.?[0] != raw.label_shape.?[0]) {
                return error.InvalidLabelShape;
            }
        },
        .gcn_infer => {
            if (raw.input_shape == null) return error.MissingModelShape;
        },
    }

    return .{
        .id = raw.id,
        .suite = suite,
        .kind = kind,
        .device = device,
        .dtype = dtype,
        .warmup_iterations = raw.warmup_iterations,
        .measured_iterations = raw.measured_iterations,
        .batch_size = raw.batch_size,
        .thread_count = raw.thread_count,
        .seed = raw.seed,
        .lhs_shape = raw.lhs_shape,
        .rhs_shape = raw.rhs_shape,
        .input_shape = raw.input_shape,
        .label_shape = raw.label_shape,
        .output_shape = raw.output_shape,
        .stride = raw.stride,
        .padding = raw.padding,
        .dilation = raw.dilation,
        .provenance = provenance,
        .notes = raw.notes,
        .pytorch_runner = raw.pytorch_runner,
        .path = path,
    };
}

fn parseSuite(value: []const u8) !Suite {
    if (std.mem.eql(u8, value, "primitive")) return .primitive;
    if (std.mem.eql(u8, value, "blas")) return .blas;
    if (std.mem.eql(u8, value, "autograd")) return .autograd;
    if (std.mem.eql(u8, value, "memory")) return .memory;
    if (std.mem.eql(u8, value, "model-train")) return .model_train;
    if (std.mem.eql(u8, value, "model-infer")) return .model_infer;
    return error.UnknownBenchmarkSuite;
}

fn parseKind(value: []const u8) !Kind {
    if (std.mem.eql(u8, value, "primitive_add")) return .primitive_add;
    if (std.mem.eql(u8, value, "primitive_matmul")) return .primitive_matmul;
    if (std.mem.eql(u8, value, "blas_dot")) return .blas_dot;
    if (std.mem.eql(u8, value, "blas_matvec")) return .blas_matvec;
    if (std.mem.eql(u8, value, "blas_conv2d_im2col")) return .blas_conv2d_im2col;
    if (std.mem.eql(u8, value, "autograd_dot_backward")) return .autograd_dot_backward;
    if (std.mem.eql(u8, value, "autograd_matvec_backward")) return .autograd_matvec_backward;
    if (std.mem.eql(u8, value, "memory_tensor_cache_cycle")) return .memory_tensor_cache_cycle;
    if (std.mem.eql(u8, value, "memory_mnist_train_step")) return .memory_mnist_train_step;
    if (std.mem.eql(u8, value, "mnist_mlp_train")) return .mnist_mlp_train;
    if (std.mem.eql(u8, value, "mnist_mlp_infer")) return .mnist_mlp_infer;
    if (std.mem.eql(u8, value, "dqn_cartpole_train")) return .dqn_cartpole_train;
    if (std.mem.eql(u8, value, "dqn_cartpole_infer")) return .dqn_cartpole_infer;
    if (std.mem.eql(u8, value, "gcn_train")) return .gcn_train;
    if (std.mem.eql(u8, value, "gcn_infer")) return .gcn_infer;
    return error.UnknownBenchmarkKind;
}

fn parseDType(value: []const u8) !DType {
    if (std.mem.eql(u8, value, "f32")) return .f32;
    return error.UnsupportedBenchmarkDType;
}

fn requireBatchedModelShapes(raw: RawSpec, require_labels: bool) !void {
    if (raw.input_shape == null) return error.MissingModelShape;
    if (raw.batch_size == null) return error.MissingBatchSize;
    if (raw.input_shape.?[0] != raw.batch_size.?) return error.InvalidBatchSize;
    if (require_labels) {
        if (raw.label_shape == null) return error.MissingModelShape;
        if (raw.label_shape.?[0] != raw.batch_size.?) return error.InvalidLabelShape;
    }
}

fn requireDotShapes(raw: RawSpec) !void {
    if (raw.lhs_shape == null or raw.rhs_shape == null) return error.MissingPrimitiveShape;
    if (raw.lhs_shape.?.len != 1 or raw.rhs_shape.?.len != 1) return error.InvalidLinearAlgebraShape;
    if (raw.lhs_shape.?[0] != raw.rhs_shape.?[0]) return error.IncompatibleLinearAlgebraShape;
}

fn requireMatvecShapes(raw: RawSpec) !void {
    if (raw.lhs_shape == null or raw.rhs_shape == null) return error.MissingPrimitiveShape;
    if (raw.lhs_shape.?.len != 2 or raw.rhs_shape.?.len != 1) return error.InvalidLinearAlgebraShape;
    if (raw.lhs_shape.?[1] != raw.rhs_shape.?[0]) return error.IncompatibleLinearAlgebraShape;
}

fn requireConv2dShapes(raw: RawSpec) !void {
    if (raw.lhs_shape == null or raw.rhs_shape == null) return error.MissingPrimitiveShape;
    if (raw.lhs_shape.?.len != 4 or raw.rhs_shape.?.len != 4) return error.InvalidConvolutionShape;
    if (raw.lhs_shape.?[1] != raw.rhs_shape.?[1]) return error.IncompatibleLinearAlgebraShape;
    if (raw.rhs_shape.?[2] != raw.rhs_shape.?[3]) return error.InvalidConvolutionShape;
    if (raw.stride == 0 or raw.dilation == 0) return error.InvalidBenchmarkShape;
}

fn validateProvenance(raw: RawBenchmarkProvenance) !result.BenchmarkProvenance {
    const data_source = std.mem.trim(u8, raw.data_source orelse return error.MissingBenchmarkDataSource, " \t\r\n");
    if (data_source.len == 0) return error.InvalidBenchmarkDataSource;

    const preprocessing = raw.preprocessing orelse return error.MissingBenchmarkPreprocessing;
    for (preprocessing) |step| {
        if (std.mem.trim(u8, step, " \t\r\n").len == 0) return error.InvalidBenchmarkPreprocessing;
    }

    return .{
        .data_source = data_source,
        .preprocessing = preprocessing,
    };
}

test "load benchmark spec from json slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "primitive.add.f32",
        \\  "suite": "primitive",
        \\  "kind": "primitive_add",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape lhs", "reshape rhs"]
        \\  },
        \\  "lhs_shape": [64, 64],
        \\  "rhs_shape": [64, 64]
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});
    const spec = try validate("inline.json", parsed);

    try std.testing.expectEqualStrings("primitive.add.f32", spec.id);
    try std.testing.expectEqual(Suite.primitive, spec.suite);
    try std.testing.expectEqual(Kind.primitive_add, spec.kind);
    try std.testing.expectEqual(DType.f32, spec.dtype);
    try std.testing.expectEqual(@as(usize, 2), spec.lhs_shape.?.len);
    try std.testing.expectEqualStrings("synthetic.splitmix64", spec.provenance.data_source);
    try std.testing.expectEqual(@as(usize, 2), spec.provenance.preprocessing.len);
}

test "load dqn benchmark spec from json slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "model-train.dqn-cartpole.synthetic.f32.batch32",
        \\  "suite": "model-train",
        \\  "kind": "dqn_cartpole_train",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape states", "derive transitions"]
        \\  },
        \\  "batch_size": 32,
        \\  "input_shape": [32, 4]
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});
    const spec = try validate("inline-dqn.json", parsed);

    try std.testing.expectEqual(Kind.dqn_cartpole_train, spec.kind);
    try std.testing.expectEqual(@as(usize, 32), spec.batch_size.?);
}

test "load autograd matvec benchmark spec from json slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "autograd.matvec-backward.f32.128x64",
        \\  "suite": "autograd",
        \\  "kind": "autograd_matvec_backward",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape matrix", "reshape vector"]
        \\  },
        \\  "lhs_shape": [128, 64],
        \\  "rhs_shape": [64]
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});
    const spec = try validate("inline-autograd-matvec.json", parsed);

    try std.testing.expectEqual(Suite.autograd, spec.suite);
    try std.testing.expectEqual(Kind.autograd_matvec_backward, spec.kind);
    try std.testing.expectEqual(@as(usize, 128), spec.lhs_shape.?[0]);
    try std.testing.expectEqual(@as(usize, 64), spec.rhs_shape.?[0]);
}

test "load memory benchmark spec from json slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "memory.tensor-cache.f32.batch4",
        \\  "suite": "memory",
        \\  "kind": "memory_tensor_cache_cycle",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape tensors", "reuse identical shapes"]
        \\  },
        \\  "batch_size": 4,
        \\  "lhs_shape": [64, 64]
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});
    const spec = try validate("inline-memory.json", parsed);

    try std.testing.expectEqual(Suite.memory, spec.suite);
    try std.testing.expectEqual(Kind.memory_tensor_cache_cycle, spec.kind);
    try std.testing.expectEqual(@as(usize, 4), spec.batch_size.?);
    try std.testing.expectEqual(@as(usize, 2), spec.lhs_shape.?.len);
}

test "load conv2d benchmark spec from json slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "blas.conv2d.f32",
        \\  "suite": "blas",
        \\  "kind": "blas_conv2d_im2col",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape input", "reshape weights", "im2col lowering parameters"]
        \\  },
        \\  "lhs_shape": [8, 1, 28, 28],
        \\  "rhs_shape": [8, 1, 3, 3],
        \\  "stride": 1,
        \\  "padding": 1,
        \\  "dilation": 1
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});
    const spec = try validate("inline-conv.json", parsed);

    try std.testing.expectEqual(Suite.blas, spec.suite);
    try std.testing.expectEqual(Kind.blas_conv2d_im2col, spec.kind);
    try std.testing.expectEqual(@as(usize, 1), spec.stride);
    try std.testing.expectEqual(@as(usize, 1), spec.padding);
    try std.testing.expectEqual(@as(usize, 1), spec.dilation);
}

test "benchmark spec requires explicit provenance" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();
    const raw =
        \\{
        \\  "id": "primitive.add.f32",
        \\  "suite": "primitive",
        \\  "kind": "primitive_add",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "lhs_shape": [64, 64],
        \\  "rhs_shape": [64, 64]
        \\}
    ;
    const parsed = try std.json.parseFromSliceLeaky(RawSpec, allocator, raw, .{});

    try std.testing.expectError(error.MissingBenchmarkProvenance, validate("inline-missing-provenance.json", parsed));
}

test "device request parsing accepts host aliases and cuda indices" {
    try std.testing.expectEqualDeep(DeviceRequest{}, try DeviceRequest.parse(null));
    try std.testing.expectEqualDeep(DeviceRequest{}, try DeviceRequest.parse("host"));
    try std.testing.expectEqualDeep(DeviceRequest{}, try DeviceRequest.parse("cpu"));
    try std.testing.expectEqualDeep(
        DeviceRequest{ .kind = .cuda, .cuda_device_index = 0 },
        try DeviceRequest.parse("cuda"),
    );
    try std.testing.expectEqualDeep(
        DeviceRequest{ .kind = .cuda, .cuda_device_index = 2 },
        try DeviceRequest.parse("cuda:2"),
    );
}

test "device request parsing rejects invalid selectors" {
    try std.testing.expectError(error.InvalidBenchmarkDevice, DeviceRequest.parse("rocm"));
    try std.testing.expectError(error.InvalidBenchmarkDevice, DeviceRequest.parse("cuda:"));
    try std.testing.expectError(error.InvalidBenchmarkDevice, DeviceRequest.parse("cuda:x"));
}
