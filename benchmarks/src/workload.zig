const std = @import("std");
const zg = @import("zigrad");
const manifest = @import("manifest.zig");
const result = @import("result.zig");
const DqnBenchmarkModel = @import("dqn_bench_model.zig").DqnBenchmarkModel;
const GcnBenchmarkModel = @import("gcn_bench_model.zig").GcnBenchmarkModel;
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
        .dqn_cartpole_train => runDqnTrain(allocator, spec),
        .dqn_cartpole_infer => runDqnInfer(allocator, spec),
        .gcn_train => runGcnTrain(allocator, spec),
        .gcn_infer => runGcnInfer(allocator, spec),
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
        try oneMnistTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            spec.input_shape.?,
            label_values,
            spec.label_shape.?,
            optimizer,
        );
    }
    for (timings) |*timing| {
        timer.reset();
        try oneMnistTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            spec.input_shape.?,
            label_values,
            spec.label_shape.?,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromMnist(allocator, spec),
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
        .shapes = try shapeMetadataFromMnist(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .notes = spec.notes,
    };
}

fn runDqnTrain(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var policy = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer policy.deinit();

    var target = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer target.deinit();
    target.setRequiresGrad(false);

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = 1e-4,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();
    try policy.attachOptimizer(optimizer);

    const state_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 31);
    defer allocator.free(state_values);
    const next_state_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 37);
    defer allocator.free(next_state_values);
    const action_values = try makeActionIndices(allocator, batch_size, 2, spec.seed +% 41);
    defer allocator.free(action_values);
    const reward_values = try makeRewardSlice(allocator, batch_size, spec.seed +% 43);
    defer allocator.free(reward_values);
    const done_values = try makeDoneSlice(allocator, batch_size, spec.seed +% 47);
    defer allocator.free(done_values);
    const gamma_values = try makeFilledSlice(allocator, batch_size, 0.99);
    defer allocator.free(gamma_values);
    const one_values = try makeFilledSlice(allocator, batch_size, 1.0);
    defer allocator.free(one_values);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try oneDqnTrainingStep(
            &graph,
            device,
            &policy,
            &target,
            state_values,
            next_state_values,
            action_values,
            reward_values,
            done_values,
            gamma_values,
            one_values,
            spec.input_shape.?,
            optimizer,
        );
    }
    for (timings) |*timing| {
        timer.reset();
        try oneDqnTrainingStep(
            &graph,
            device,
            &policy,
            &target,
            state_values,
            next_state_values,
            action_values,
            reward_values,
            done_values,
            gamma_values,
            one_values,
            spec.input_shape.?,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromDqnTrain(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .notes = spec.notes,
    };
}

fn runDqnInfer(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var model = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();
    model.setRequiresGrad(false);

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 53);
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
        .shapes = try shapeMetadataFromDqnInfer(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .notes = spec.notes,
    };
}

fn runGcnTrain(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const input_shape = spec.input_shape.?;
    const label_shape = spec.label_shape.?;
    const node_count = input_shape[0];
    const edge_values = try makeGraphEdgeIndex(allocator, node_count, 4);
    defer allocator.free(edge_values);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var model = try GcnBenchmarkModel(f32).init(
        allocator,
        device,
        &graph,
        input_shape[1],
        label_shape[1],
        spec.seed,
    );
    defer model.deinit();

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = 0.01,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();
    try model.attachOptimizer(optimizer);

    const input_values = try makeDeterministicSlice(allocator, countElements(input_shape), spec.seed +% 59);
    defer allocator.free(input_values);
    const label_values = try makeOneHotLabels(allocator, node_count, label_shape[1], spec.seed +% 61);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try oneGcnTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            edge_values,
            optimizer,
        );
    }
    for (timings) |*timing| {
        timer.reset();
        try oneGcnTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            edge_values,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromGcn(allocator, spec, edge_values.len / 2, true),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = node_count,
        .throughput_unit = "nodes",
        .notes = spec.notes,
    };
}

fn runGcnInfer(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const input_shape = spec.input_shape.?;
    const node_count = input_shape[0];
    const edge_values = try makeGraphEdgeIndex(allocator, node_count, 4);
    defer allocator.free(edge_values);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var timer = try std.time.Timer.start();
    var model = try GcnBenchmarkModel(f32).init(
        allocator,
        device,
        &graph,
        input_shape[1],
        7,
        spec.seed,
    );
    defer model.deinit();
    model.setRequiresGrad(false);

    const input_values = try makeDeterministicSlice(allocator, countElements(input_shape), spec.seed +% 67);
    defer allocator.free(input_values);

    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = &graph });
    defer input.deinit();
    const edge_index = try zg.NDTensor(usize).from_slice(device, edge_values, &.{ 2, edge_values.len / 2 }, .{
        .graph = &graph,
    });
    defer edge_index.deinit();
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    for (0..spec.warmup_iterations) |_| {
        const output = try model.forward(input, edge_index);
        output.deinit();
    }
    for (timings) |*timing| {
        timer.reset();
        const output = try model.forward(input, edge_index);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromGcn(allocator, spec, edge_values.len / 2, false),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = node_count,
        .throughput_unit = "nodes",
        .notes = spec.notes,
    };
}

fn oneMnistTrainingStep(
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

fn oneDqnTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    policy: *DqnBenchmarkModel(f32),
    target: *DqnBenchmarkModel(f32),
    state_values: []const f32,
    next_state_values: []const f32,
    action_values: []const usize,
    reward_values: []const f32,
    done_values: []const f32,
    gamma_values: []const f32,
    one_values: []const f32,
    input_shape: []const usize,
    optimizer: zg.Optimizer,
) !void {
    const Tensor = zg.NDTensor(f32);
    const batch_size = input_shape[0];

    const states = try Tensor.from_slice(device, state_values, input_shape, .{ .graph = graph });
    defer states.deinit();
    const next_states = try Tensor.from_slice(device, next_state_values, input_shape, .{ .graph = graph });
    defer next_states.deinit();
    const actions = try zg.NDTensor(usize).from_slice(device, action_values, &.{ batch_size, 1 }, .{
        .graph = graph,
    });
    defer actions.deinit();
    const rewards = try Tensor.from_slice(device, reward_values, &.{ batch_size, 1 }, .{ .graph = graph });
    defer rewards.deinit();
    const dones = try Tensor.from_slice(device, done_values, &.{ batch_size, 1 }, .{ .graph = graph });
    defer dones.deinit();
    const gamma = try Tensor.from_slice(device, gamma_values, &.{ batch_size, 1 }, .{ .graph = graph });
    defer gamma.deinit();
    const ones = try Tensor.from_slice(device, one_values, &.{ batch_size, 1 }, .{ .graph = graph });
    defer ones.deinit();

    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const next_q_values = try target.forward(next_states);
    defer next_q_values.deinit();
    const max_next_q_values = try next_q_values.max_along(.{ .dim = 1, .keep_dims = true });
    defer max_next_q_values.deinit();

    const discounted = try max_next_q_values.mul(gamma);
    defer discounted.deinit();
    const not_done = try ones.sub(dones);
    defer not_done.deinit();
    const masked = try discounted.mul(not_done);
    defer masked.deinit();
    const targets = try rewards.add(masked);
    defer targets.deinit();

    zg.runtime.grad_enabled = true;
    const all_q_values = try policy.forward(states);
    defer all_q_values.deinit();
    const selected_q_values = try all_q_values.gather(actions.data, 1);
    defer selected_q_values.deinit();

    const loss = try zg.loss.smooth_l1_loss(f32, selected_q_values, targets, 1.0);
    defer loss.deinit();

    try loss.backward();
    try optimizer.step();
    policy.zeroGrad();
}

fn oneGcnTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *GcnBenchmarkModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
    edge_values: []const usize,
    optimizer: zg.Optimizer,
) !void {
    const Tensor = zg.NDTensor(f32);

    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    defer input.deinit();
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    defer labels.deinit();
    const edge_index = try zg.NDTensor(usize).from_slice(device, edge_values, &.{ 2, edge_values.len / 2 }, .{
        .graph = graph,
    });
    defer edge_index.deinit();

    const output = try model.forward(input, edge_index);
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

fn makeFilledSlice(
    allocator: std.mem.Allocator,
    count: usize,
    value: f32,
) ![]f32 {
    const values = try allocator.alloc(f32, count);
    for (values) |*slot| slot.* = value;
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

fn makeActionIndices(
    allocator: std.mem.Allocator,
    batch_size: usize,
    action_count: usize,
    seed: u64,
) ![]usize {
    const values = try allocator.alloc(usize, batch_size);
    for (values, 0..) |*value, row| {
        value.* = @as(usize, @intCast(splitmix64(seed +% @as(u64, row)) % action_count));
    }
    return values;
}

fn makeRewardSlice(
    allocator: std.mem.Allocator,
    batch_size: usize,
    seed: u64,
) ![]f32 {
    const values = try makeDeterministicSlice(allocator, batch_size, seed);
    for (values) |*value| value.* *= 0.5;
    return values;
}

fn makeDoneSlice(
    allocator: std.mem.Allocator,
    batch_size: usize,
    seed: u64,
) ![]f32 {
    const values = try allocator.alloc(f32, batch_size);
    for (values, 0..) |*value, row| {
        const mixed = splitmix64(seed +% @as(u64, row));
        value.* = if (mixed % 7 == 0) 1.0 else 0.0;
    }
    return values;
}

fn makeGraphEdgeIndex(
    allocator: std.mem.Allocator,
    node_count: usize,
    fanout: usize,
) ![]usize {
    const edge_count = node_count * fanout;
    const values = try allocator.alloc(usize, edge_count * 2);

    for (0..node_count) |node| {
        for (0..fanout) |slot| {
            const edge_index = (node * fanout) + slot;
            values[edge_index] = node;
            values[edge_count + edge_index] = switch (slot) {
                0 => node,
                1 => if (node == 0) node_count - 1 else node - 1,
                2 => (node + 1) % node_count,
                else => (node + 2) % node_count,
            };
        }
    }

    return values;
}

fn shapeMetadataFromPrimitive(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 2);
    shapes[0] = .{ .name = "lhs", .dims = spec.lhs_shape.? };
    shapes[1] = .{ .name = "rhs", .dims = spec.rhs_shape.? };
    return shapes;
}

fn shapeMetadataFromMnist(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const label_shape_count: usize = if (spec.label_shape == null) 0 else 1;
    const shapes = try allocator.alloc(result.ShapeMetadata, 1 + label_shape_count);
    shapes[0] = .{ .name = "input", .dims = spec.input_shape.? };
    if (spec.label_shape) |label_shape| {
        shapes[1] = .{ .name = "labels", .dims = label_shape };
    }
    return shapes;
}

fn shapeMetadataFromDqnTrain(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const batch_size = spec.batch_size.?;
    const shapes = try allocator.alloc(result.ShapeMetadata, 5);
    shapes[0] = .{ .name = "state", .dims = spec.input_shape.? };
    shapes[1] = .{ .name = "next_state", .dims = spec.input_shape.? };
    shapes[2] = .{ .name = "action", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    shapes[3] = .{ .name = "reward", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    shapes[4] = .{ .name = "done", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    return shapes;
}

fn shapeMetadataFromDqnInfer(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 1);
    shapes[0] = .{ .name = "state", .dims = spec.input_shape.? };
    return shapes;
}

fn shapeMetadataFromGcn(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    edge_count: usize,
    include_labels: bool,
) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, if (include_labels) 3 else 2);
    shapes[0] = .{ .name = "node_features", .dims = spec.input_shape.? };
    shapes[1] = .{ .name = "edge_index", .dims = try allocDims(allocator, &.{ 2, edge_count }) };
    if (include_labels) {
        shapes[2] = .{ .name = "labels", .dims = spec.label_shape.? };
    }
    return shapes;
}

fn allocDims(allocator: std.mem.Allocator, dims: []const usize) ![]const usize {
    return try allocator.dupe(usize, dims);
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

test "run dqn infer benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 4 };
    const spec: manifest.Spec = .{
        .id = "test.dqn.infer",
        .suite = .model_infer,
        .kind = .dqn_cartpole_infer,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 1), output.shapes.len);
    try std.testing.expectEqual(@as(usize, 8), output.batch_size.?);
}

test "run gcn train benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 16, 8 };
    const label_shape = [_]usize{ 16, 7 };
    const spec: manifest.Spec = .{
        .id = "test.gcn.train",
        .suite = .model_train,
        .kind = .gcn_train,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .label_shape = label_shape[0..],
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 3), output.shapes.len);
    try std.testing.expectEqualStrings("edge_index", output.shapes[1].name);
}
