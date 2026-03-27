const std = @import("std");

pub const Suite = enum {
    primitive,
    blas,
    autograd,
    model_train,
    model_infer,

    pub fn asString(self: Suite) []const u8 {
        return switch (self) {
            .primitive => "primitive",
            .blas => "blas",
            .autograd => "autograd",
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
    autograd_dot_backward,
    autograd_matvec_backward,
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
            .autograd_dot_backward => "autograd_dot_backward",
            .autograd_matvec_backward => "autograd_matvec_backward",
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

pub const Spec = struct {
    id: []const u8,
    suite: Suite,
    kind: Kind,
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
    notes: ?[]const u8 = null,
    pytorch_runner: ?[]const u8 = null,
    path: []const u8,
};

const RawSpec = struct {
    id: []const u8,
    suite: []const u8,
    kind: []const u8,
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
    const dtype = try parseDType(raw.dtype);

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
        .notes = raw.notes,
        .pytorch_runner = raw.pytorch_runner,
        .path = path,
    };
}

fn parseSuite(value: []const u8) !Suite {
    if (std.mem.eql(u8, value, "primitive")) return .primitive;
    if (std.mem.eql(u8, value, "blas")) return .blas;
    if (std.mem.eql(u8, value, "autograd")) return .autograd;
    if (std.mem.eql(u8, value, "model-train")) return .model_train;
    if (std.mem.eql(u8, value, "model-infer")) return .model_infer;
    return error.UnknownBenchmarkSuite;
}

fn parseKind(value: []const u8) !Kind {
    if (std.mem.eql(u8, value, "primitive_add")) return .primitive_add;
    if (std.mem.eql(u8, value, "primitive_matmul")) return .primitive_matmul;
    if (std.mem.eql(u8, value, "blas_dot")) return .blas_dot;
    if (std.mem.eql(u8, value, "blas_matvec")) return .blas_matvec;
    if (std.mem.eql(u8, value, "autograd_dot_backward")) return .autograd_dot_backward;
    if (std.mem.eql(u8, value, "autograd_matvec_backward")) return .autograd_matvec_backward;
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
