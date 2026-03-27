const std = @import("std");
const zg = @import("zigrad");
const manifest = @import("manifest.zig");
const result = @import("result.zig");
const metadata = @import("metadata.zig");
const MnistBenchmarkModel = @import("mnist_bench_model.zig").MnistBenchmarkModel;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub const RunOutput = struct {
    shapes: []const result.ShapeMetadata,
    batch_size: ?usize,
    setup_latency_ns: u64,
    timings_ns: []const u64,
    throughput_items: ?usize = null,
    throughput_unit: ?[]const u8 = null,
    notes: ?[]const u8 = null,
};

pub fn applyThreadCount(thread_count: ?u32) void {
    const count = thread_count orelse return;
    var value_buf: [32:0]u8 = undefined;
    const value = std.fmt.bufPrintZ(&value_buf, "{d}", .{count}) catch return;

    setEnv("OMP_NUM_THREADS", value);
    setEnv("OPENBLAS_NUM_THREADS", value);
    setEnv("MKL_NUM_THREADS", value);
    setEnv("BLIS_NUM_THREADS", value);
    setEnv("VECLIB_MAXIMUM_THREADS", value);
}

pub fn run(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    applyThreadCount(spec.thread_count);

    return switch (spec.kind) {
        .primitive_add => runPrimitiveAdd(allocator, spec),
        .primitive_matmul => runPrimitiveMatmul(allocator, spec),
        .mnist_mlp_train => runMnistTrain(allocator, spec),
        .mnist_mlp_infer => runMnistInfer(allocator, spec),
    };
}

fn runPrimitiveAdd(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(lhs_data);
    const rhs_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(rhs_data);

    var timer = try std.time.Timer.start();
    const lhs = try Tensor.from_slice(device, lhs_data, spec.lhs_shape.?, .{ .graph = &graph });
    defer lhs.deinit();
    const rhs = try Tensor.from_slice(device, rhs_data, spec.rhs_shape.?, .{ .graph = &graph });
    defer rhs.deinit();
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const output = try lhs.add(rhs);
        output.deinit();
    }
    for (timings) |*timing| {
        timer.reset();
        const output = try lhs.add(rhs);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromPrimitive(allocator, spec),
        .batch_size = spec.batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = countElements(spec.lhs_shape.?),
        .throughput_unit = "elements",
        .notes = spec.notes,
    };
}

fn runPrimitiveMatmul(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(lhs_data);
    const rhs_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(rhs_data);

    var timer = try std.time.Timer.start();
    const lhs = try Tensor.from_slice(device, lhs_data, spec.lhs_shape.?, .{ .graph = &graph });
    defer lhs.deinit();
    const rhs = try Tensor.from_slice(device, rhs_data, spec.rhs_shape.?, .{ .graph = &graph });
    defer rhs.deinit();
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const output = try lhs.bmm(rhs, .{});
        output.deinit();
    }
    for (timings) |*timing| {
        timer.reset();
        const output = try lhs.bmm(rhs, .{});
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromPrimitive(allocator, spec),
        .batch_size = spec.batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = spec.batch_size,
        .throughput_unit = null,
        .notes = spec.notes,
    };
}

fn runMnistTrain(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var model = try MnistBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    var sgd = zg.optim.SGD.init(allocator, .{
        .lr = 0.01,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
    });
    defer sgd.deinit();
    const optimizer = sgd.optimizer();
    try model.attachOptimizer(optimizer);

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 11);
    defer allocator.free(input_values);
    const label_values = try makeOneHotLabels(allocator, batch_size, 10, spec.seed +% 17);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try oneTrainingStep(&graph, device, &model, input_values, spec.input_shape.?, label_values, spec.label_shape.?, optimizer);
    }
    for (timings) |*timing| {
        timer.reset();
        try oneTrainingStep(&graph, device, &model, input_values, spec.input_shape.?, label_values, spec.label_shape.?, optimizer);
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromModel(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .notes = spec.notes,
    };
}

fn runMnistInfer(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var model = try MnistBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();
    model.setRequiresGrad(false);

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 23);
    defer allocator.free(input_values);

    const input = try Tensor.from_slice(device, input_values, spec.input_shape.?, .{ .graph = &graph });
    defer input.deinit();
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    for (0..spec.warmup_iterations) |_| {
        const output = try model.forward(input);
        output.deinit();
    }
    for (timings) |*timing| {
        timer.reset();
        const output = try model.forward(input);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromModel(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .notes = spec.notes,
    };
}

fn oneTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *MnistBenchmarkModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
    optimizer: zg.Optimizer,
) !void {
    const Tensor = zg.NDTensor(f32);
    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    defer input.deinit();
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    defer labels.deinit();

    const output = try model.forward(input);
    defer output.deinit();

    const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, labels);
    defer loss.deinit();

    try loss.backward();
    try optimizer.step();
    model.zeroGrad();
}

fn countElements(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| total *= dim;
    return total;
}

fn makeDeterministicSlice(
    allocator: std.mem.Allocator,
    count: usize,
    seed: u64,
) ![]f32 {
    const values = try allocator.alloc(f32, count);
    for (values, 0..) |*value, index| {
        const mixed = splitmix64(seed +% @as(u64, index));
        const normalized = (@as(f64, @floatFromInt(mixed % 5000)) / 5000.0) - 0.5;
        value.* = @as(f32, @floatCast(normalized));
    }
    return values;
}

fn makeOneHotLabels(
    allocator: std.mem.Allocator,
    batch_size: usize,
    classes: usize,
    seed: u64,
) ![]f32 {
    const values = try allocator.alloc(f32, batch_size * classes);
    @memset(values, 0);
    for (0..batch_size) |row| {
        const class_index = @as(usize, @intCast(splitmix64(seed +% @as(u64, row)) % classes));
        values[(row * classes) + class_index] = 1;
    }
    return values;
}

fn shapeMetadataFromPrimitive(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 2);
    shapes[0] = .{ .name = "lhs", .dims = spec.lhs_shape.? };
    shapes[1] = .{ .name = "rhs", .dims = spec.rhs_shape.? };
    return shapes;
}

fn shapeMetadataFromModel(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const label_shape_count: usize = if (spec.label_shape == null) 0 else 1;
    const shapes = try allocator.alloc(result.ShapeMetadata, 1 + label_shape_count);
    shapes[0] = .{ .name = "input", .dims = spec.input_shape.? };
    if (spec.label_shape) |label_shape| {
        shapes[1] = .{ .name = "labels", .dims = label_shape };
    }
    return shapes;
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn setEnv(name: []const u8, value: [:0]const u8) void {
    var name_buf: [64:0]u8 = undefined;
    const name_z = std.fmt.bufPrintZ(&name_buf, "{s}", .{name}) catch return;
    _ = setenv(name_z.ptr, value.ptr, 1);
}
