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
    memory: ?result.MemoryStats = null,
    host_blas_telemetry: ?result.HostBlasTelemetry = null,
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

fn resetHostBenchmarkTelemetry(host: *zg.device.HostDevice) void {
    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
}

fn captureHostBlasTelemetry(host: *const zg.device.HostDevice) result.HostBlasTelemetry {
    const op = host.opTelemetry();
    const dispatch = host.dispatchTelemetry();
    return .{
        .dot_calls = op.dot_calls,
        .matvec_calls = op.matvec_calls,
        .matmul_calls = op.matmul_calls,
        .bmm_acc_calls = op.bmm_acc_calls,
        .direct_bmm_dispatches = dispatch.direct_bmm_dispatches,
        .fallback_bmm_dispatches = dispatch.fallback_bmm_dispatches,
        .fallback_bmm_batches = dispatch.fallback_bmm_batches,
    };
}

pub fn run(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    applyThreadCount(spec.thread_count);

    return switch (spec.kind) {
        .primitive_add => runPrimitiveAdd(allocator, spec),
        .primitive_matmul => runPrimitiveMatmul(allocator, spec),
        .blas_dot => runBlasDot(allocator, spec),
        .blas_matvec => runBlasMatvec(allocator, spec),
        .blas_conv2d_im2col => runBlasConv2dIm2col(allocator, spec),
        .autograd_dot_backward => runAutogradDotBackward(allocator, spec),
        .autograd_matvec_backward => runAutogradMatvecBackward(allocator, spec),
        .memory_tensor_cache_cycle => runMemoryTensorCacheCycle(allocator, spec),
        .memory_mnist_train_step => runMemoryMnistTrainStep(allocator, spec),
        .mnist_mlp_train => runMnistTrain(allocator, spec),
        .mnist_mlp_infer => runMnistInfer(allocator, spec),
        .dqn_cartpole_train => runDqnTrain(allocator, spec),
        .dqn_cartpole_infer => runDqnInfer(allocator, spec),
        .gcn_train => runGcnTrain(allocator, spec),
        .gcn_infer => runGcnInfer(allocator, spec),
    };
}

fn runBlasDot(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Array = zg.NDArray(f32);
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(lhs_data);
    const rhs_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(rhs_data);

    var timer = try std.time.Timer.start();
    var lhs = try Array.from_slice(lhs_data, spec.lhs_shape.?, device);
    defer lhs.deinit(device);
    var rhs = try Array.from_slice(rhs_data, spec.rhs_shape.?, device);
    defer rhs.deinit(device);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        var output = try lhs.dot(rhs, device);
        output.deinit(device);
    }
    resetHostBenchmarkTelemetry(&host);
    for (timings) |*timing| {
        timer.reset();
        var output = try lhs.dot(rhs, device);
        timing.* = timer.read();
        output.deinit(device);
    }

    return .{
        .shapes = try shapeMetadataFromVectorPair(allocator, spec),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = countElements(spec.lhs_shape.?),
        .throughput_unit = "elements",
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runBlasMatvec(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Array = zg.NDArray(f32);
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const matrix_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(matrix_data);
    const vector_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(vector_data);

    var timer = try std.time.Timer.start();
    var matrix = try Array.from_slice(matrix_data, spec.lhs_shape.?, device);
    defer matrix.deinit(device);
    var vector = try Array.from_slice(vector_data, spec.rhs_shape.?, device);
    defer vector.deinit(device);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        var output = try matrix.matvec(vector, device, .{});
        output.deinit(device);
    }
    resetHostBenchmarkTelemetry(&host);
    for (timings) |*timing| {
        timer.reset();
        var output = try matrix.matvec(vector, device, .{});
        timing.* = timer.read();
        output.deinit(device);
    }

    return .{
        .shapes = try shapeMetadataFromMatrixVector(allocator, spec),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = countElements(spec.lhs_shape.?),
        .throughput_unit = "matrix-elements",
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runBlasConv2dIm2col(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const Array = zg.NDArray(f32);
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(input_values);
    const weight_values = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(weight_values);

    const output_shape = try zg.conv_utils.conv2dOutputShape(spec.lhs_shape.?, spec.rhs_shape.?, .{
        .stride = spec.stride,
        .padding = spec.padding,
        .dilation = spec.dilation,
    });

    var timer = try std.time.Timer.start();
    var input = try Array.from_slice(input_values, spec.lhs_shape.?, device);
    defer input.deinit(device);
    var weights = try Array.from_slice(weight_values, spec.rhs_shape.?, device);
    defer weights.deinit(device);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        var output = try zg.conv_utils.conv2dForwardIm2col(f32, input, weights, null, .{
            .stride = spec.stride,
            .padding = spec.padding,
            .dilation = spec.dilation,
        }, device);
        output.deinit(device);
    }
    resetHostBenchmarkTelemetry(&host);
    for (timings) |*timing| {
        timer.reset();
        var output = try zg.conv_utils.conv2dForwardIm2col(f32, input, weights, null, .{
            .stride = spec.stride,
            .padding = spec.padding,
            .dilation = spec.dilation,
        }, device);
        timing.* = timer.read();
        output.deinit(device);
    }

    return .{
        .shapes = try shapeMetadataFromConv2d(allocator, spec, output_shape[0..]),
        .batch_size = spec.lhs_shape.?[0],
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = spec.lhs_shape.?[0],
        .throughput_unit = "samples",
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runAutogradDotBackward(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(lhs_data);
    const rhs_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(rhs_data);

    var timer = try std.time.Timer.start();
    try prepareAutogradDotOperands(allocator, device, lhs_data, rhs_data, spec.lhs_shape.?, spec.rhs_shape.?);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneAutogradDotBackwardStep(allocator, device, lhs_data, rhs_data, spec.lhs_shape.?, spec.rhs_shape.?);
    }
    resetHostBenchmarkTelemetry(&host);
    for (timings) |*timing| {
        timer.reset();
        try oneAutogradDotBackwardStep(allocator, device, lhs_data, rhs_data, spec.lhs_shape.?, spec.rhs_shape.?);
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromVectorPair(allocator, spec),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = countElements(spec.lhs_shape.?),
        .throughput_unit = "elements",
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runAutogradMatvecBackward(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const matrix_data = try makeDeterministicSlice(allocator, countElements(spec.lhs_shape.?), spec.seed);
    defer allocator.free(matrix_data);
    const vector_data = try makeDeterministicSlice(allocator, countElements(spec.rhs_shape.?), spec.seed +% 1);
    defer allocator.free(vector_data);

    var timer = try std.time.Timer.start();
    try prepareAutogradMatvecOperands(allocator, device, matrix_data, vector_data, spec.lhs_shape.?, spec.rhs_shape.?);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneAutogradMatvecBackwardStep(allocator, device, matrix_data, vector_data, spec.lhs_shape.?, spec.rhs_shape.?);
    }
    resetHostBenchmarkTelemetry(&host);
    for (timings) |*timing| {
        timer.reset();
        try oneAutogradMatvecBackwardStep(allocator, device, matrix_data, vector_data, spec.lhs_shape.?, spec.rhs_shape.?);
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromMatrixVector(allocator, spec),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = countElements(spec.lhs_shape.?),
        .throughput_unit = "matrix-elements",
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runMemoryTensorCacheCycle(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
    const buffer_shape = spec.lhs_shape.?;
    const retained_buffers = spec.batch_size.?;

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const buffer_values = try makeDeterministicSlice(allocator, countElements(buffer_shape), spec.seed);
    defer allocator.free(buffer_values);

    var timer = try std.time.Timer.start();
    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneMemoryTensorCacheCycle(allocator, device, buffer_values, buffer_shape, retained_buffers);
    }

    host.resetCacheTelemetry();
    resetHostBenchmarkTelemetry(&host);

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (timings) |*timing| {
        timer.reset();
        try oneMemoryTensorCacheCycle(allocator, device, buffer_values, buffer_shape, retained_buffers);
        timing.* = timer.read();
    }

    const telemetry = host.cacheTelemetry();

    return .{
        .shapes = try shapeMetadataFromMemoryBuffer(allocator, spec),
        .batch_size = retained_buffers,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = retained_buffers * countElements(buffer_shape),
        .throughput_unit = "elements",
        .memory = .{
            .peak_live_bytes = @as(u64, @intCast(telemetry.peak_live_bytes)),
            .final_live_bytes = @as(u64, @intCast(telemetry.live_bytes)),
            .peak_scratch_bytes = @as(u64, @intCast(telemetry.peak_scratch_bytes)),
        },
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
        .notes = spec.notes,
    };
}

fn runMemoryMnistTrainStep(allocator: std.mem.Allocator, spec: manifest.Spec) !RunOutput {
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

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 71);
    defer allocator.free(input_values);
    const label_values = try makeOneHotLabels(allocator, batch_size, spec.label_shape.?[1], spec.seed +% 73);
    defer allocator.free(label_values);
    const setup_latency_ns = timer.read();

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

    host.resetCacheTelemetry();
    resetHostBenchmarkTelemetry(&host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

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
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    const telemetry = host.cacheTelemetry();
    const final_graph_arena_bytes = graph.queryArenaCapacityBytes();

    return .{
        .shapes = try shapeMetadataFromMnist(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = .{
            .peak_live_bytes = @as(u64, @intCast(telemetry.peak_live_bytes)),
            .final_live_bytes = @as(u64, @intCast(telemetry.live_bytes)),
            .peak_graph_arena_bytes = @as(u64, @intCast(peak_graph_arena_bytes)),
            .final_graph_arena_bytes = @as(u64, @intCast(final_graph_arena_bytes)),
            .peak_scratch_bytes = @as(u64, @intCast(telemetry.peak_scratch_bytes)),
        },
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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
    resetHostBenchmarkTelemetry(&host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(&host),
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

fn oneMemoryTensorCacheCycle(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    buffer_values: []const f32,
    buffer_shape: []const usize,
    retained_buffers: usize,
) !void {
    const Array = zg.NDArray(f32);
    const arrays = try allocator.alloc(Array, retained_buffers);
    defer allocator.free(arrays);

    var initialized: usize = 0;
    errdefer {
        while (initialized > 0) {
            initialized -= 1;
            arrays[initialized].deinit(device);
        }
    }

    for (0..retained_buffers) |index| {
        arrays[index] = try Array.from_slice(buffer_values, buffer_shape, device);
        initialized += 1;
    }

    while (initialized > 0) {
        initialized -= 1;
        arrays[initialized].deinit(device);
    }
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

fn prepareAutogradDotOperands(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    lhs_values: []const f32,
    rhs_values: []const f32,
    lhs_shape: []const usize,
    rhs_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };
    const lhs = try Tensor.from_slice(device, lhs_values, lhs_shape, opts);
    defer lhs.deinit();
    const rhs = try Tensor.from_slice(device, rhs_values, rhs_shape, opts);
    defer rhs.deinit();
    try lhs.setup_grad(0);
    try rhs.setup_grad(0);
}

fn oneAutogradDotBackwardStep(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    lhs_values: []const f32,
    rhs_values: []const f32,
    lhs_shape: []const usize,
    rhs_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };
    const lhs = try Tensor.from_slice(device, lhs_values, lhs_shape, opts);
    defer lhs.deinit();
    const rhs = try Tensor.from_slice(device, rhs_values, rhs_shape, opts);
    defer rhs.deinit();

    const output = try lhs.dot(rhs);
    defer output.deinit();
    try output.backward();
}

fn prepareAutogradMatvecOperands(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    matrix_values: []const f32,
    vector_values: []const f32,
    matrix_shape: []const usize,
    vector_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };
    const matrix = try Tensor.from_slice(device, matrix_values, matrix_shape, opts);
    defer matrix.deinit();
    const vector = try Tensor.from_slice(device, vector_values, vector_shape, opts);
    defer vector.deinit();
    try matrix.setup_grad(0);
    try vector.setup_grad(0);
}

fn oneAutogradMatvecBackwardStep(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    matrix_values: []const f32,
    vector_values: []const f32,
    matrix_shape: []const usize,
    vector_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };
    const matrix = try Tensor.from_slice(device, matrix_values, matrix_shape, opts);
    defer matrix.deinit();
    const vector = try Tensor.from_slice(device, vector_values, vector_shape, opts);
    defer vector.deinit();

    const output = try matrix.matvec(vector, .{});
    defer output.deinit();
    try output.backward();
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

fn shapeMetadataFromVectorPair(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 2);
    shapes[0] = .{ .name = "lhs", .dims = spec.lhs_shape.? };
    shapes[1] = .{ .name = "rhs", .dims = spec.rhs_shape.? };
    return shapes;
}

fn shapeMetadataFromMatrixVector(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 2);
    shapes[0] = .{ .name = "matrix", .dims = spec.lhs_shape.? };
    shapes[1] = .{ .name = "vector", .dims = spec.rhs_shape.? };
    return shapes;
}

fn shapeMetadataFromConv2d(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    output_shape: []const usize,
) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 3);
    shapes[0] = .{ .name = "input", .dims = spec.lhs_shape.? };
    shapes[1] = .{ .name = "weights", .dims = spec.rhs_shape.? };
    shapes[2] = .{ .name = "output", .dims = try allocDims(allocator, output_shape) };
    return shapes;
}

fn shapeMetadataFromMemoryBuffer(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 1);
    shapes[0] = .{ .name = "buffer", .dims = spec.lhs_shape.? };
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

fn inlineProvenance(preprocessing: []const []const u8) result.BenchmarkProvenance {
    return .{
        .data_source = "synthetic.splitmix64",
        .preprocessing = preprocessing,
    };
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
        .provenance = inlineProvenance(&.{"reshape states to input_shape"}),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 1), output.shapes.len);
    try std.testing.expectEqual(@as(usize, 8), output.batch_size.?);
}

test "run blas dot benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const lhs_shape = [_]usize{16};
    const rhs_shape = [_]usize{16};
    const spec: manifest.Spec = .{
        .id = "test.blas.dot",
        .suite = .blas,
        .kind = .blas_dot,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .thread_count = 1,
        .seed = 1,
        .lhs_shape = lhs_shape[0..],
        .rhs_shape = rhs_shape[0..],
        .provenance = inlineProvenance(&.{ "reshape lhs to lhs_shape", "reshape rhs to rhs_shape" }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expectEqualStrings("lhs", output.shapes[0].name);
}

test "run autograd matvec benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const matrix_shape = [_]usize{ 8, 4 };
    const vector_shape = [_]usize{4};
    const spec: manifest.Spec = .{
        .id = "test.autograd.matvec",
        .suite = .autograd,
        .kind = .autograd_matvec_backward,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .thread_count = 1,
        .seed = 1,
        .lhs_shape = matrix_shape[0..],
        .rhs_shape = vector_shape[0..],
        .provenance = inlineProvenance(&.{ "reshape matrix to lhs_shape", "reshape vector to rhs_shape" }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expectEqualStrings("matrix", output.shapes[0].name);
}

test "run conv2d im2col benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 2, 1, 5, 5 };
    const weight_shape = [_]usize{ 3, 1, 3, 3 };
    const spec: manifest.Spec = .{
        .id = "test.blas.conv2d",
        .suite = .blas,
        .kind = .blas_conv2d_im2col,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .thread_count = 1,
        .seed = 1,
        .lhs_shape = input_shape[0..],
        .rhs_shape = weight_shape[0..],
        .stride = 1,
        .padding = 1,
        .dilation = 1,
        .provenance = inlineProvenance(&.{
            "reshape input to lhs_shape",
            "reshape weights to rhs_shape",
            "apply declared stride, padding, and dilation",
        }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 3), output.shapes.len);
    try std.testing.expectEqualStrings("output", output.shapes[2].name);
    try std.testing.expectEqual(@as(usize, 2), output.batch_size.?);
}

test "run memory tensor cache benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const buffer_shape = [_]usize{ 8, 8 };
    const spec: manifest.Spec = .{
        .id = "test.memory.tensor-cache",
        .suite = .memory,
        .kind = .memory_tensor_cache_cycle,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 4,
        .thread_count = 1,
        .seed = 1,
        .lhs_shape = buffer_shape[0..],
        .provenance = inlineProvenance(&.{
            "reshape tensors to lhs_shape",
            "reuse identical allocation sizes across each cache cycle",
        }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 1), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_live_bytes.? > 0);
    try std.testing.expectEqual(@as(u64, 0), output.memory.?.final_live_bytes.?);
}

test "run memory mnist training benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 4, 1, 28, 28 };
    const label_shape = [_]usize{ 4, 10 };
    const spec: manifest.Spec = .{
        .id = "test.memory.mnist-train",
        .suite = .memory,
        .kind = .memory_mnist_train_step,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 4,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .label_shape = label_shape[0..],
        .provenance = inlineProvenance(&.{
            "reshape inputs to input_shape",
            "derive one-hot labels from deterministic class pattern",
        }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_live_bytes.? >= output.memory.?.final_live_bytes.?);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
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
        .provenance = inlineProvenance(&.{
            "reshape node features to input_shape",
            "generate deterministic ring-plus-skip edge_index",
            "derive one-hot node labels from deterministic class pattern",
        }),
        .path = "inline",
    };

    const output = try run(arena.allocator(), spec);
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 3), output.shapes.len);
    try std.testing.expectEqualStrings("edge_index", output.shapes[1].name);
}
