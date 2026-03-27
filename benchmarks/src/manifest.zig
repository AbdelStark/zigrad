const std = @import("std");

pub const Suite = enum {
    primitive,
    model_train,
    model_infer,

    pub fn asString(self: Suite) []const u8 {
        return switch (self) {
            .primitive => "primitive",
            .model_train => "model-train",
            .model_infer => "model-infer",
        };
    }
};

pub const Kind = enum {
    primitive_add,
    primitive_matmul,
    mnist_mlp_train,
    mnist_mlp_infer,

    pub fn asString(self: Kind) []const u8 {
        return switch (self) {
            .primitive_add => "primitive_add",
            .primitive_matmul => "primitive_matmul",
            .mnist_mlp_train => "mnist_mlp_train",
            .mnist_mlp_infer => "mnist_mlp_infer",
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
        .mnist_mlp_train => {
            if (raw.input_shape == null or raw.label_shape == null) {
                return error.MissingModelShape;
            }
            if (raw.batch_size == null) return error.MissingBatchSize;
        },
        .mnist_mlp_infer => {
            if (raw.input_shape == null) return error.MissingModelShape;
            if (raw.batch_size == null) return error.MissingBatchSize;
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
    if (std.mem.eql(u8, value, "model-train")) return .model_train;
    if (std.mem.eql(u8, value, "model-infer")) return .model_infer;
    return error.UnknownBenchmarkSuite;
}

fn parseKind(value: []const u8) !Kind {
    if (std.mem.eql(u8, value, "primitive_add")) return .primitive_add;
    if (std.mem.eql(u8, value, "primitive_matmul")) return .primitive_matmul;
    if (std.mem.eql(u8, value, "mnist_mlp_train")) return .mnist_mlp_train;
    if (std.mem.eql(u8, value, "mnist_mlp_infer")) return .mnist_mlp_infer;
    return error.UnknownBenchmarkKind;
}

fn parseDType(value: []const u8) !DType {
    if (std.mem.eql(u8, value, "f32")) return .f32;
    return error.UnsupportedBenchmarkDType;
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
