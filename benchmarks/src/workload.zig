const std = @import("std");
const zg = @import("zigrad");
const manifest = @import("manifest.zig");
const metadata = @import("metadata.zig");
const corridor = @import("examples_corridor_environment");
const pendulum_data = @import("examples_pendulum_dataset");
const result = @import("result.zig");
const SatoshiLmModel = @import("examples_satoshi_lm_model").SatoshiLmModel;
const CorridorControlModel = @import("examples_corridor_model").CorridorControlModel;
const DqnBenchmarkModel = @import("dqn_bench_model.zig").DqnBenchmarkModel;
const GcnBenchmarkModel = @import("gcn_bench_model.zig").GcnBenchmarkModel;
const MnistBenchmarkModel = @import("mnist_bench_model.zig").MnistBenchmarkModel;
const PendulumDynamicsModel = @import("examples_pendulum_model").PendulumDynamicsModel;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

pub const RunOutput = struct {
    shapes: []const result.ShapeMetadata,
    batch_size: ?usize,
    setup_latency_ns: u64,
    timings_ns: []const u64,
    throughput_items: ?usize = null,
    throughput_unit: ?[]const u8 = null,
    memory: ?result.MemoryStats = null,
    interop: ?result.InteropMetrics = null,
    host_blas_telemetry: ?result.HostBlasTelemetry = null,
    notes: ?[]const u8 = null,
};

pub const RunResult = struct {
    status: result.Status,
    backend: result.BackendMetadata,
    output: ?RunOutput = null,
    notes: ?[]const u8 = null,
};

const RunContext = struct {
    runtime_device: zg.device.RuntimeDevice,
    thread_count: ?u32,

    fn init(spec: manifest.Spec) anyerror!RunContext {
        var runtime_device = try zg.device.initRuntimeDevice(.{
            .kind = switch (spec.device.kind) {
                .host => .host,
                .cuda => .cuda,
            },
            .cuda_device_index = spec.device.cuda_device_index,
        }, .{
            .allow_cuda = true,
        });
        errdefer runtime_device.deinit();

        return .{
            .runtime_device = runtime_device,
            .thread_count = spec.thread_count,
        };
    }

    fn deinit(self: *RunContext) void {
        self.runtime_device.deinit();
    }

    fn host(self: *RunContext) ?*zg.device.HostDevice {
        return switch (self.runtime_device.kind) {
            .host => if (self.runtime_device.host) |*value| value else null,
            .cuda => null,
        };
    }

    fn device(self: *RunContext) zg.DeviceReference {
        return self.runtime_device.reference();
    }

    fn backend(
        self: *RunContext,
        allocator: std.mem.Allocator,
        host_blas_telemetry: ?result.HostBlasTelemetry,
    ) !result.BackendMetadata {
        return metadata.runtimeBackend(
            allocator,
            &self.runtime_device,
            self.thread_count,
            host_blas_telemetry,
        );
    }
};

pub fn expectedBatchSize(spec: manifest.Spec) ?usize {
    return switch (spec.kind) {
        .blas_conv2d_im2col => spec.lhs_shape.?[0],
        .memory_tensor_cache_cycle,
        .memory_mnist_train_step,
        .compiler_mnist_mlp_capture,
        .compiler_satoshi_lm_capture,
        .compiler_pendulum_dynamics_capture,
        .compiler_corridor_control_capture,
        .compiler_dqn_cartpole_capture,
        .mnist_mlp_train,
        .mnist_mlp_infer,
        .satoshi_lm_train,
        .satoshi_lm_infer,
        .pendulum_dynamics_train,
        .pendulum_dynamics_infer,
        .corridor_control_train,
        .corridor_control_infer,
        .dqn_cartpole_train,
        .dqn_cartpole_infer,
        => spec.batch_size,
        else => null,
    };
}

pub fn expectedShapeMetadata(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
) ![]const result.ShapeMetadata {
    return switch (spec.kind) {
        .primitive_add,
        .primitive_matmul,
        => shapeMetadataFromPrimitive(allocator, spec),
        .blas_dot,
        .autograd_dot_backward,
        => shapeMetadataFromVectorPair(allocator, spec),
        .blas_matvec,
        .autograd_matvec_backward,
        => shapeMetadataFromMatrixVector(allocator, spec),
        .blas_conv2d_im2col => blk: {
            const output_shape = try zg.conv_utils.conv2dOutputShape(spec.lhs_shape.?, spec.rhs_shape.?, .{
                .stride = spec.stride,
                .padding = spec.padding,
                .dilation = spec.dilation,
            });
            break :blk try shapeMetadataFromConv2d(allocator, spec, output_shape[0..]);
        },
        .memory_tensor_cache_cycle => shapeMetadataFromMemoryBuffer(allocator, spec),
        .memory_mnist_train_step,
        .compiler_mnist_mlp_capture,
        .mnist_mlp_train,
        .mnist_mlp_infer,
        => shapeMetadataFromMnist(allocator, spec),
        .interop_mnist_mlp_safetensors_export,
        .interop_mnist_mlp_safetensors_import,
        => shapeMetadataFromMnistCheckpoint(allocator),
        .interop_satoshi_lm_safetensors_export,
        .interop_satoshi_lm_safetensors_import,
        => shapeMetadataFromSatoshiLmCheckpoint(allocator),
        .compiler_satoshi_lm_capture,
        .satoshi_lm_train,
        .satoshi_lm_infer,
        => shapeMetadataFromSatoshiLm(allocator, spec),
        .interop_pendulum_dynamics_safetensors_export,
        .interop_pendulum_dynamics_safetensors_import,
        => shapeMetadataFromPendulumCheckpoint(allocator),
        .pendulum_dynamics_train,
        .pendulum_dynamics_infer,
        .compiler_pendulum_dynamics_capture,
        => shapeMetadataFromPendulum(allocator, spec),
        .interop_corridor_control_safetensors_export,
        .interop_corridor_control_safetensors_import,
        => shapeMetadataFromCorridorCheckpoint(allocator),
        .corridor_control_train => shapeMetadataFromCorridorTrain(allocator, spec),
        .corridor_control_infer => shapeMetadataFromCorridorInfer(allocator, spec),
        .compiler_corridor_control_capture => shapeMetadataFromCorridorTrain(allocator, spec),
        .compiler_dqn_cartpole_capture => shapeMetadataFromDqnTrain(allocator, spec),
        .interop_dqn_cartpole_safetensors_export,
        .interop_dqn_cartpole_safetensors_import,
        => shapeMetadataFromDqnCheckpoint(allocator),
        .interop_gcn_safetensors_export,
        .interop_gcn_safetensors_import,
        => shapeMetadataFromGcnCheckpoint(allocator),
        .compiler_gcn_capture => shapeMetadataFromGcn(allocator, spec, spec.input_shape.?[0] * 4, true),
        .dqn_cartpole_train => shapeMetadataFromDqnTrain(allocator, spec),
        .dqn_cartpole_infer => shapeMetadataFromDqnInfer(allocator, spec),
        .gcn_train => shapeMetadataFromGcn(allocator, spec, spec.input_shape.?[0] * 4, true),
        .gcn_infer => shapeMetadataFromGcn(allocator, spec, spec.input_shape.?[0] * 4, false),
    };
}

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

fn maybeResetHostBenchmarkTelemetry(host: ?*zg.device.HostDevice) void {
    if (host) |value| resetHostBenchmarkTelemetry(value);
}

fn maybeCaptureHostBlasTelemetry(host: ?*zg.device.HostDevice) ?result.HostBlasTelemetry {
    if (host) |value| return captureHostBlasTelemetry(value);
    return null;
}

pub fn run(allocator: std.mem.Allocator, spec: manifest.Spec) !RunResult {
    applyThreadCount(spec.thread_count);

    if (spec.device.kind != .host and (spec.suite == .memory or spec.suite == .interop)) {
        return .{
            .status = .skipped,
            .backend = try metadata.requestedBackend(allocator, spec.device, spec.thread_count),
            .notes = try std.fmt.allocPrint(
                allocator,
                "Benchmark suite `{s}` currently supports only host execution.",
                .{spec.suite.asString()},
            ),
        };
    }

    var context = RunContext.init(spec) catch |err| {
        const backend = try metadata.requestedBackend(allocator, spec.device, spec.thread_count);
        return .{
            .status = switch (err) {
                error.CudaUnsupported,
                error.CudaNotEnabled,
                error.CudaUnavailable,
                => .skipped,
                else => .failed,
            },
            .backend = backend,
            .notes = try runtimeInitFailureNote(allocator, spec, err),
        };
    };
    defer context.deinit();

    const output = switch (spec.kind) {
        .primitive_add => runPrimitiveAdd(allocator, spec, &context),
        .primitive_matmul => runPrimitiveMatmul(allocator, spec, &context),
        .blas_dot => runBlasDot(allocator, spec, &context),
        .blas_matvec => runBlasMatvec(allocator, spec, &context),
        .blas_conv2d_im2col => runBlasConv2dIm2col(allocator, spec, &context),
        .autograd_dot_backward => runAutogradDotBackward(allocator, spec, &context),
        .autograd_matvec_backward => runAutogradMatvecBackward(allocator, spec, &context),
        .memory_tensor_cache_cycle => runMemoryTensorCacheCycle(allocator, spec, &context),
        .memory_mnist_train_step => runMemoryMnistTrainStep(allocator, spec, &context),
        .compiler_mnist_mlp_capture => runCompilerMnistCapture(allocator, spec, &context),
        .compiler_satoshi_lm_capture => runCompilerSatoshiLmCapture(allocator, spec, &context),
        .compiler_pendulum_dynamics_capture => runCompilerPendulumCapture(allocator, spec, &context),
        .compiler_corridor_control_capture => runCompilerCorridorCapture(allocator, spec, &context),
        .compiler_dqn_cartpole_capture => runCompilerDqnCapture(allocator, spec, &context),
        .compiler_gcn_capture => runCompilerGcnCapture(allocator, spec, &context),
        .interop_mnist_mlp_safetensors_export => runInteropMnistSafetensorsExport(allocator, spec, &context),
        .interop_mnist_mlp_safetensors_import => runInteropMnistSafetensorsImport(allocator, spec, &context),
        .interop_satoshi_lm_safetensors_export => runInteropSatoshiLmSafetensorsExport(allocator, spec, &context),
        .interop_satoshi_lm_safetensors_import => runInteropSatoshiLmSafetensorsImport(allocator, spec, &context),
        .interop_pendulum_dynamics_safetensors_export => runInteropPendulumSafetensorsExport(allocator, spec, &context),
        .interop_pendulum_dynamics_safetensors_import => runInteropPendulumSafetensorsImport(allocator, spec, &context),
        .interop_corridor_control_safetensors_export => runInteropCorridorSafetensorsExport(allocator, spec, &context),
        .interop_corridor_control_safetensors_import => runInteropCorridorSafetensorsImport(allocator, spec, &context),
        .interop_dqn_cartpole_safetensors_export => runInteropDqnSafetensorsExport(allocator, spec, &context),
        .interop_dqn_cartpole_safetensors_import => runInteropDqnSafetensorsImport(allocator, spec, &context),
        .interop_gcn_safetensors_export => runInteropGcnSafetensorsExport(allocator, spec, &context),
        .interop_gcn_safetensors_import => runInteropGcnSafetensorsImport(allocator, spec, &context),
        .mnist_mlp_train => runMnistTrain(allocator, spec, &context),
        .mnist_mlp_infer => runMnistInfer(allocator, spec, &context),
        .satoshi_lm_train => runSatoshiLmTrain(allocator, spec, &context),
        .satoshi_lm_infer => runSatoshiLmInfer(allocator, spec, &context),
        .pendulum_dynamics_train => runPendulumTrain(allocator, spec, &context),
        .pendulum_dynamics_infer => runPendulumInfer(allocator, spec, &context),
        .corridor_control_train => runCorridorTrain(allocator, spec, &context),
        .corridor_control_infer => runCorridorInfer(allocator, spec, &context),
        .dqn_cartpole_train => runDqnTrain(allocator, spec, &context),
        .dqn_cartpole_infer => runDqnInfer(allocator, spec, &context),
        .gcn_train => runGcnTrain(allocator, spec, &context),
        .gcn_infer => runGcnInfer(allocator, spec, &context),
    } catch |err| {
        return .{
            .status = .failed,
            .backend = try context.backend(allocator, null),
            .notes = try std.fmt.allocPrint(
                allocator,
                "Benchmark execution failed: {s}",
                .{@errorName(err)},
            ),
        };
    };

    return .{
        .status = .ok,
        .backend = try context.backend(allocator, output.host_blas_telemetry),
        .output = output,
        .notes = output.notes,
    };
}

fn runtimeInitFailureNote(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    err: anyerror,
) ![]const u8 {
    return switch (err) {
        error.CudaUnsupported => try std.fmt.allocPrint(
            allocator,
            "Benchmark device `{s}` is not supported by this workload.",
            .{spec.device.kind.asString()},
        ),
        error.CudaNotEnabled => "CUDA benchmarks require a build compiled with -Denable_cuda=true.",
        error.CudaUnavailable => "CUDA benchmarks were requested, but no CUDA devices were detected.",
        error.InvalidCudaDeviceIndex => try std.fmt.allocPrint(
            allocator,
            "CUDA device {d} was requested, but it is not available on this machine.",
            .{spec.device.cuda_device_index},
        ),
        else => try std.fmt.allocPrint(
            allocator,
            "Benchmark runtime initialization failed: {s}",
            .{@errorName(err)},
        ),
    };
}

fn runBlasDot(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Array = zg.NDArray(f32);
    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runBlasMatvec(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Array = zg.NDArray(f32);
    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runBlasConv2dIm2col(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Array = zg.NDArray(f32);
    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runAutogradDotBackward(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runAutogradMatvecBackward(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runPrimitiveAdd(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runPrimitiveMatmul(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runMemoryTensorCacheCycle(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const buffer_shape = spec.lhs_shape.?;
    const retained_buffers = spec.batch_size.?;

    const host = context.host() orelse unreachable;
    const device = context.device();

    const buffer_values = try makeDeterministicSlice(allocator, countElements(buffer_shape), spec.seed);
    defer allocator.free(buffer_values);

    var timer = try std.time.Timer.start();
    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneMemoryTensorCacheCycle(allocator, device, buffer_values, buffer_shape, retained_buffers);
    }

    host.resetCacheTelemetry();
    resetHostBenchmarkTelemetry(host);

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
        .host_blas_telemetry = captureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runMemoryMnistTrainStep(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host() orelse unreachable;
    const device = context.device();

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
    resetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = captureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerMnistCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try MnistBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    const input_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 101);
    defer allocator.free(input_values);
    const label_values = try makeOneHotLabels(allocator, batch_size, spec.label_shape.?[1], spec.seed +% 103);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerMnistCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            spec.input_shape.?,
            label_values,
            spec.label_shape.?,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerMnistCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            spec.input_shape.?,
            label_values,
            spec.label_shape.?,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromMnist(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerSatoshiLmCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;
    const input_shape = spec.input_shape.?;
    const label_shape = spec.label_shape.?;
    const context_len = input_shape[1];
    const vocab_size = input_shape[2];

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try SatoshiLmModel(f32).initWithGraphAndSeed(
        device,
        context_len,
        vocab_size,
        charLmHiddenSize(vocab_size),
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const token_stream = try makeSatoshiLmTokenStream(allocator, batch_size, context_len, vocab_size, spec.seed +% 107);
    defer allocator.free(token_stream);
    const input_values = try makeSatoshiLmInputBatch(allocator, token_stream, batch_size, context_len, vocab_size);
    defer allocator.free(input_values);
    const label_values = try makeSatoshiLmLabelBatch(allocator, token_stream, batch_size, context_len, vocab_size);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerSatoshiLmCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerSatoshiLmCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromSatoshiLm(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerPendulumCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try PendulumDynamicsModel(f32).initWithGraphAndSeed(
        device,
        pendulum_data.input_feature_count,
        pendulumHiddenSize(),
        pendulum_data.output_feature_count,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    var generated = try pendulum_data.generateTransitions(allocator, batch_size, spec.seed +% 101, .{});
    defer generated.deinit(allocator);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerPendulumCaptureStep(
            &graph,
            device,
            &model,
            generated.inputs,
            spec.input_shape.?,
            generated.targets,
            spec.label_shape.?,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerPendulumCaptureStep(
            &graph,
            device,
            &model,
            generated.inputs,
            spec.input_shape.?,
            generated.targets,
            spec.label_shape.?,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromPendulum(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerCorridorCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var policy = try CorridorControlModel(f32).initWithGraphAndSeed(
        device,
        corridor.state_feature_count,
        corridorHiddenSize(),
        corridor.action_count,
        &graph,
        spec.seed,
    );
    defer policy.deinit();

    var target = try policy.clone();
    defer target.deinit();
    target.set_requires_grad(false);

    var generated = try corridor.generateSyntheticTransitionBatch(allocator, batch_size, spec.seed +% 163, .{});
    defer generated.deinit(allocator);
    const action_mask_values = try makeActionMask(allocator, generated.actions, corridor.action_count);
    defer allocator.free(action_mask_values);
    const gamma_values = try makeFilledSlice(allocator, batch_size, 0.97);
    defer allocator.free(gamma_values);
    const one_values = try makeFilledSlice(allocator, batch_size, 1.0);
    defer allocator.free(one_values);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerCorridorCaptureStep(
            &graph,
            device,
            &policy,
            &target,
            generated.states,
            generated.next_states,
            action_mask_values,
            generated.rewards,
            generated.dones,
            gamma_values,
            one_values,
            spec.input_shape.?,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerCorridorCaptureStep(
            &graph,
            device,
            &policy,
            &target,
            generated.states,
            generated.next_states,
            action_mask_values,
            generated.rewards,
            generated.dones,
            gamma_values,
            one_values,
            spec.input_shape.?,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromCorridorTrain(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerDqnCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var policy = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer policy.deinit();

    var target = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer target.deinit();
    target.setRequiresGrad(false);

    const state_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 109);
    defer allocator.free(state_values);
    const next_state_values = try makeDeterministicSlice(allocator, countElements(spec.input_shape.?), spec.seed +% 113);
    defer allocator.free(next_state_values);
    const action_values = try makeActionIndices(allocator, batch_size, 2, spec.seed +% 127);
    defer allocator.free(action_values);
    const action_mask_values = try makeActionMask(allocator, action_values, 2);
    defer allocator.free(action_mask_values);
    const reward_values = try makeRewardSlice(allocator, batch_size, spec.seed +% 131);
    defer allocator.free(reward_values);
    const done_values = try makeDoneSlice(allocator, batch_size, spec.seed +% 137);
    defer allocator.free(done_values);
    const gamma_values = try makeFilledSlice(allocator, batch_size, 0.99);
    defer allocator.free(gamma_values);
    const one_values = try makeFilledSlice(allocator, batch_size, 1.0);
    defer allocator.free(one_values);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerDqnCaptureStep(
            &graph,
            device,
            &policy,
            &target,
            state_values,
            next_state_values,
            action_mask_values,
            reward_values,
            done_values,
            gamma_values,
            one_values,
            spec.input_shape.?,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerDqnCaptureStep(
            &graph,
            device,
            &policy,
            &target,
            state_values,
            next_state_values,
            action_mask_values,
            reward_values,
            done_values,
            gamma_values,
            one_values,
            spec.input_shape.?,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromDqnTrain(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCompilerGcnCapture(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const input_shape = spec.input_shape.?;
    const label_shape = spec.label_shape.?;
    const node_count = input_shape[0];
    const edge_values = try makeGraphEdgeIndex(allocator, node_count, 4);
    defer allocator.free(edge_values);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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

    const input_values = try makeDeterministicSlice(allocator, countElements(input_shape), spec.seed +% 139);
    defer allocator.free(input_values);
    const label_values = try makeOneHotLabels(allocator, node_count, label_shape[1], spec.seed +% 149);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();

    for (0..spec.warmup_iterations) |_| {
        try oneCompilerGcnCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            edge_values,
        );
    }

    if (host) |value| value.resetCacheTelemetry();
    maybeResetHostBenchmarkTelemetry(host);
    var peak_graph_arena_bytes = graph.queryArenaCapacityBytes();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (timings) |*timing| {
        timer.reset();
        try oneCompilerGcnCaptureStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            edge_values,
        );
        timing.* = timer.read();
        peak_graph_arena_bytes = @max(peak_graph_arena_bytes, graph.queryArenaCapacityBytes());
    }

    return .{
        .shapes = try shapeMetadataFromGcn(allocator, spec, edge_values.len / 2, true),
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = node_count,
        .throughput_unit = "nodes",
        .memory = compilerCaptureMemoryStats(host, peak_graph_arena_bytes, graph.queryArenaCapacityBytes()),
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runInteropMnistSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try MnistBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    const sample_checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromMnistCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropMnistSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try MnistBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneAffineCheckpointImport(MnistBenchmarkModel(f32), io_allocator, device, checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneAffineCheckpointImport(MnistBenchmarkModel(f32), io_allocator, device, checkpoint);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromMnistCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropSatoshiLmSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const context_len = interopSatoshiLmContextLen();
    const vocab_size = interopSatoshiLmVocabSize();
    const hidden_size = charLmHiddenSize(vocab_size);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try SatoshiLmModel(f32).initWithGraphAndSeed(
        device,
        context_len,
        vocab_size,
        hidden_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const sample_checkpoint = try serializeSatoshiLmCheckpoint(io_allocator, &model);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeSatoshiLmCheckpoint(io_allocator, &model);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeSatoshiLmCheckpoint(io_allocator, &model);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromSatoshiLmCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropSatoshiLmSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const context_len = interopSatoshiLmContextLen();
    const vocab_size = interopSatoshiLmVocabSize();
    const hidden_size = charLmHiddenSize(vocab_size);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try SatoshiLmModel(f32).initWithGraphAndSeed(
        device,
        context_len,
        vocab_size,
        hidden_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const checkpoint = try serializeSatoshiLmCheckpoint(io_allocator, &model);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneSatoshiLmCheckpointImport(io_allocator, device, checkpoint, context_len, vocab_size, hidden_size);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneSatoshiLmCheckpointImport(io_allocator, device, checkpoint, context_len, vocab_size, hidden_size);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromSatoshiLmCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropPendulumSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const input_size = pendulum_data.input_feature_count;
    const hidden_size = pendulumHiddenSize();
    const output_size = pendulum_data.output_feature_count;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try PendulumDynamicsModel(f32).initWithGraphAndSeed(
        device,
        input_size,
        hidden_size,
        output_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const sample_checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromPendulumCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropPendulumSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const input_size = pendulum_data.input_feature_count;
    const hidden_size = pendulumHiddenSize();
    const output_size = pendulum_data.output_feature_count;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try PendulumDynamicsModel(f32).initWithGraphAndSeed(
        device,
        input_size,
        hidden_size,
        output_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try onePendulumCheckpointImport(io_allocator, device, checkpoint, input_size, hidden_size, output_size);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try onePendulumCheckpointImport(io_allocator, device, checkpoint, input_size, hidden_size, output_size);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromPendulumCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropCorridorSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const input_size = corridor.state_feature_count;
    const hidden_size = corridorHiddenSize();
    const output_size = corridor.action_count;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try CorridorControlModel(f32).initWithGraphAndSeed(
        device,
        input_size,
        hidden_size,
        output_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const sample_checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromCorridorCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropCorridorSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;
    const input_size = corridor.state_feature_count;
    const hidden_size = corridorHiddenSize();
    const output_size = corridor.action_count;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try CorridorControlModel(f32).initWithGraphAndSeed(
        device,
        input_size,
        hidden_size,
        output_size,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneCorridorCheckpointImport(io_allocator, device, checkpoint, input_size, hidden_size, output_size);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneCorridorCheckpointImport(io_allocator, device, checkpoint, input_size, hidden_size, output_size);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromCorridorCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropDqnSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    const sample_checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromDqnCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropDqnSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try DqnBenchmarkModel(f32).init(allocator, device, &graph, spec.seed);
    defer model.deinit();

    const checkpoint = try serializeAffineCheckpoint(io_allocator, model.weights[0..], model.biases[0..]);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneAffineCheckpointImport(DqnBenchmarkModel(f32), io_allocator, device, checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneAffineCheckpointImport(DqnBenchmarkModel(f32), io_allocator, device, checkpoint);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromDqnCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropGcnSafetensorsExport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try GcnBenchmarkModel(f32).init(
        allocator,
        device,
        &graph,
        interopGcnInputFeatureCount(),
        interopGcnOutputFeatureCount(),
        spec.seed,
    );
    defer model.deinit();

    const sample_checkpoint = try serializeGcnCheckpoint(io_allocator, &model);
    defer io_allocator.free(sample_checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        const checkpoint = try serializeGcnCheckpoint(io_allocator, &model);
        io_allocator.free(checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const checkpoint = try serializeGcnCheckpoint(io_allocator, &model);
        timing.* = timer.read();
        io_allocator.free(checkpoint);
    }

    return interopCheckpointOutput(
        try shapeMetadataFromGcnCheckpoint(allocator),
        setup_latency_ns,
        timings,
        sample_checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runInteropGcnSafetensorsImport(
    allocator: std.mem.Allocator,
    spec: manifest.Spec,
    context: *RunContext,
) !RunOutput {
    const host = context.host();
    const device = context.device();
    const io_allocator = std.heap.page_allocator;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var timer = try std.time.Timer.start();
    var model = try GcnBenchmarkModel(f32).init(
        allocator,
        device,
        &graph,
        interopGcnInputFeatureCount(),
        interopGcnOutputFeatureCount(),
        spec.seed,
    );
    defer model.deinit();

    const checkpoint = try serializeGcnCheckpoint(io_allocator, &model);
    defer io_allocator.free(checkpoint);
    const setup_latency_ns = timer.read();

    const timings = try allocator.alloc(u64, spec.measured_iterations);
    for (0..spec.warmup_iterations) |_| {
        try oneGcnCheckpointImport(io_allocator, device, checkpoint);
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneGcnCheckpointImport(io_allocator, device, checkpoint);
        timing.* = timer.read();
    }

    return interopCheckpointOutput(
        try shapeMetadataFromGcnCheckpoint(allocator),
        setup_latency_ns,
        timings,
        checkpoint.len,
        maybeCaptureHostBlasTelemetry(host),
        spec.notes,
    );
}

fn runMnistTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runMnistInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runSatoshiLmTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;
    const input_shape = spec.input_shape.?;
    const label_shape = spec.label_shape.?;
    const context_len = input_shape[1];
    const vocab_size = input_shape[2];

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try SatoshiLmModel(f32).initWithGraphAndSeed(
        device,
        context_len,
        vocab_size,
        charLmHiddenSize(vocab_size),
        &graph,
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
    try model.attach_optimizer(optimizer);

    const token_stream = try makeSatoshiLmTokenStream(allocator, batch_size, context_len, vocab_size, spec.seed +% 71);
    defer allocator.free(token_stream);
    const input_values = try makeSatoshiLmInputBatch(allocator, token_stream, batch_size, context_len, vocab_size);
    defer allocator.free(input_values);
    const label_values = try makeSatoshiLmLabelBatch(allocator, token_stream, batch_size, context_len, vocab_size);
    defer allocator.free(label_values);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try oneSatoshiLmTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            optimizer,
        );
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneSatoshiLmTrainingStep(
            &graph,
            device,
            &model,
            input_values,
            input_shape,
            label_values,
            label_shape,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromSatoshiLm(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runSatoshiLmInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;
    const input_shape = spec.input_shape.?;
    const context_len = input_shape[1];
    const vocab_size = input_shape[2];

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try SatoshiLmModel(f32).initWithGraphAndSeed(
        device,
        context_len,
        vocab_size,
        charLmHiddenSize(vocab_size),
        &graph,
        spec.seed,
    );
    defer model.deinit();
    model.set_requires_grad(false);

    const token_stream = try makeSatoshiLmTokenStream(allocator, batch_size, context_len, vocab_size, spec.seed +% 83);
    defer allocator.free(token_stream);
    const input_values = try makeSatoshiLmInputBatch(allocator, token_stream, batch_size, context_len, vocab_size);
    defer allocator.free(input_values);

    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = &graph });
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
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const output = try model.forward(input);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromSatoshiLm(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runPendulumTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try PendulumDynamicsModel(f32).initWithGraphAndSeed(
        device,
        pendulum_data.input_feature_count,
        pendulumHiddenSize(),
        pendulum_data.output_feature_count,
        &graph,
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
    try model.attach_optimizer(optimizer);

    var generated = try pendulum_data.generateTransitions(allocator, batch_size, spec.seed +% 89, .{});
    defer generated.deinit(allocator);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try onePendulumTrainingStep(
            &graph,
            device,
            &model,
            generated.inputs,
            spec.input_shape.?,
            generated.targets,
            spec.label_shape.?,
            optimizer,
        );
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try onePendulumTrainingStep(
            &graph,
            device,
            &model,
            generated.inputs,
            spec.input_shape.?,
            generated.targets,
            spec.label_shape.?,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromPendulum(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runPendulumInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try PendulumDynamicsModel(f32).initWithGraphAndSeed(
        device,
        pendulum_data.input_feature_count,
        pendulumHiddenSize(),
        pendulum_data.output_feature_count,
        &graph,
        spec.seed,
    );
    defer model.deinit();
    model.set_requires_grad(false);

    var generated = try pendulum_data.generateTransitions(allocator, batch_size, spec.seed +% 97, .{});
    defer generated.deinit(allocator);

    const input = try Tensor.from_slice(device, generated.inputs, spec.input_shape.?, .{ .graph = &graph });
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
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const output = try model.forward(input);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromPendulum(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCorridorTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try CorridorControlModel(f32).initWithGraphAndSeed(
        device,
        corridor.state_feature_count,
        corridorHiddenSize(),
        corridor.action_count,
        &graph,
        spec.seed,
    );
    defer model.deinit();

    var target = try model.clone();
    defer target.deinit();
    target.set_requires_grad(false);

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = 0.004,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();
    try model.attach_optimizer(optimizer);

    var generated = try corridor.generateSyntheticTransitionBatch(allocator, batch_size, spec.seed +% 151, .{});
    defer generated.deinit(allocator);
    const gamma_values = try makeFilledSlice(allocator, batch_size, 0.97);
    defer allocator.free(gamma_values);
    const one_values = try makeFilledSlice(allocator, batch_size, 1.0);
    defer allocator.free(one_values);

    const setup_latency_ns = timer.read();
    const timings = try allocator.alloc(u64, spec.measured_iterations);

    for (0..spec.warmup_iterations) |_| {
        try oneCorridorTrainingStep(
            &graph,
            device,
            &model,
            &target,
            generated.states,
            generated.next_states,
            generated.actions,
            generated.rewards,
            generated.dones,
            gamma_values,
            one_values,
            spec.input_shape.?,
            optimizer,
        );
    }
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        try oneCorridorTrainingStep(
            &graph,
            device,
            &model,
            &target,
            generated.states,
            generated.next_states,
            generated.actions,
            generated.rewards,
            generated.dones,
            gamma_values,
            one_values,
            spec.input_shape.?,
            optimizer,
        );
        timing.* = timer.read();
    }

    return .{
        .shapes = try shapeMetadataFromCorridorTrain(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runCorridorInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

    var timer = try std.time.Timer.start();
    var model = try CorridorControlModel(f32).initWithGraphAndSeed(
        device,
        corridor.state_feature_count,
        corridorHiddenSize(),
        corridor.action_count,
        &graph,
        spec.seed,
    );
    defer model.deinit();
    model.set_requires_grad(false);

    const input_values = try corridor.generateSyntheticStates(allocator, batch_size, spec.seed +% 157, .{});
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
    maybeResetHostBenchmarkTelemetry(host);
    for (timings) |*timing| {
        timer.reset();
        const output = try model.forward(input);
        timing.* = timer.read();
        output.deinit();
    }

    return .{
        .shapes = try shapeMetadataFromCorridorInfer(allocator, spec),
        .batch_size = batch_size,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings,
        .throughput_items = batch_size,
        .throughput_unit = "samples",
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runDqnTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runDqnInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const batch_size = spec.batch_size.?;

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runGcnTrain(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const input_shape = spec.input_shape.?;
    const label_shape = spec.label_shape.?;
    const node_count = input_shape[0];
    const edge_values = try makeGraphEdgeIndex(allocator, node_count, 4);
    defer allocator.free(edge_values);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
        .notes = spec.notes,
    };
}

fn runGcnInfer(allocator: std.mem.Allocator, spec: manifest.Spec, context: *RunContext) !RunOutput {
    const Tensor = zg.NDTensor(f32);
    const input_shape = spec.input_shape.?;
    const node_count = input_shape[0];
    const edge_values = try makeGraphEdgeIndex(allocator, node_count, 4);
    defer allocator.free(edge_values);

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    const host = context.host();
    const device = context.device();

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
    maybeResetHostBenchmarkTelemetry(host);
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
        .host_blas_telemetry = maybeCaptureHostBlasTelemetry(host),
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

fn oneSatoshiLmTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *SatoshiLmModel(f32),
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
    model.zero_grad();
}

fn onePendulumTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *PendulumDynamicsModel(f32),
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

    const loss = try zg.loss.mse_loss(f32, output, labels);
    defer loss.deinit();

    try loss.backward();
    try optimizer.step();
    model.zero_grad();
}

fn oneCompilerPendulumCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *PendulumDynamicsModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);

    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    const output = try model.forward(input);
    const loss = try zg.loss.mse_loss(f32, output, labels);
    graph.teardown(&loss.node);
}

fn oneCompilerMnistCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *MnistBenchmarkModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    const output = try model.forward(input);
    const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, labels);
    graph.teardown(&loss.node);
}

fn oneCompilerSatoshiLmCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *SatoshiLmModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    const output = try model.forward(input);
    const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, labels);
    graph.teardown(&loss.node);
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

fn oneCorridorTrainingStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    policy: *CorridorControlModel(f32),
    target: *CorridorControlModel(f32),
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
    policy.zero_grad();
    try target.soft_update_from(policy, 0.08);
}

fn oneCompilerCorridorCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    policy: *CorridorControlModel(f32),
    target: *CorridorControlModel(f32),
    state_values: []const f32,
    next_state_values: []const f32,
    action_mask_values: []const f32,
    reward_values: []const f32,
    done_values: []const f32,
    gamma_values: []const f32,
    one_values: []const f32,
    input_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const batch_size = input_shape[0];
    const selector_values = [_]f32{ 1.0, 1.0, 1.0 };

    const states = try Tensor.from_slice(device, state_values, input_shape, .{ .graph = graph });
    const next_states = try Tensor.from_slice(device, next_state_values, input_shape, .{ .graph = graph });
    const action_mask = try Tensor.from_slice(device, action_mask_values, &.{ batch_size, corridor.action_count }, .{ .graph = graph });
    const rewards = try Tensor.from_slice(device, reward_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const dones = try Tensor.from_slice(device, done_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const gamma = try Tensor.from_slice(device, gamma_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const ones = try Tensor.from_slice(device, one_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const selector = try Tensor.from_slice(device, &selector_values, &.{ corridor.action_count, 1 }, .{ .graph = graph });

    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const next_q_values = try target.forward(next_states);
    const max_next_q_values = try next_q_values.max_along(.{ .dim = 1, .keep_dims = true });
    const discounted = try max_next_q_values.mul(gamma);
    const not_done = try ones.sub(dones);
    const masked = try discounted.mul(not_done);
    const targets = try rewards.add(masked);

    zg.runtime.grad_enabled = true;
    const all_q_values = try policy.forward(states);
    const masked_q_values = try all_q_values.mul(action_mask);
    const selected_q_values = try masked_q_values.bmm(selector, .{});
    const loss = try zg.loss.smooth_l1_loss(f32, selected_q_values, targets, 1.0);
    graph.teardown(&loss.node);
}

fn oneCompilerDqnCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    policy: *DqnBenchmarkModel(f32),
    target: *DqnBenchmarkModel(f32),
    state_values: []const f32,
    next_state_values: []const f32,
    action_mask_values: []const f32,
    reward_values: []const f32,
    done_values: []const f32,
    gamma_values: []const f32,
    one_values: []const f32,
    input_shape: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const batch_size = input_shape[0];
    const selector_values = [_]f32{ 1.0, 1.0 };

    const states = try Tensor.from_slice(device, state_values, input_shape, .{ .graph = graph });
    const next_states = try Tensor.from_slice(device, next_state_values, input_shape, .{ .graph = graph });
    const action_mask = try Tensor.from_slice(device, action_mask_values, &.{ batch_size, 2 }, .{ .graph = graph });
    const rewards = try Tensor.from_slice(device, reward_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const dones = try Tensor.from_slice(device, done_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const gamma = try Tensor.from_slice(device, gamma_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const ones = try Tensor.from_slice(device, one_values, &.{ batch_size, 1 }, .{ .graph = graph });
    const selector = try Tensor.from_slice(device, &selector_values, &.{ 2, 1 }, .{ .graph = graph });

    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const next_q_values = try target.forward(next_states);
    const max_next_q_values = try next_q_values.max_along(.{ .dim = 1, .keep_dims = true });
    const discounted = try max_next_q_values.mul(gamma);
    const not_done = try ones.sub(dones);
    const masked = try discounted.mul(not_done);
    const targets = try rewards.add(masked);

    zg.runtime.grad_enabled = true;
    const all_q_values = try policy.forward(states);
    const masked_q_values = try all_q_values.mul(action_mask);
    const selected_q_values = try masked_q_values.bmm(selector, .{});
    const loss = try zg.loss.smooth_l1_loss(f32, selected_q_values, targets, 1.0);
    graph.teardown(&loss.node);
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

fn oneCompilerGcnCaptureStep(
    graph: *zg.Graph,
    device: zg.DeviceReference,
    model: *GcnBenchmarkModel(f32),
    input_values: []const f32,
    input_shape: []const usize,
    label_values: []const f32,
    label_shape: []const usize,
    edge_values: []const usize,
) !void {
    const Tensor = zg.NDTensor(f32);

    const input = try Tensor.from_slice(device, input_values, input_shape, .{ .graph = graph });
    const labels = try Tensor.from_slice(device, label_values, label_shape, .{ .graph = graph });
    const edge_index = try zg.NDTensor(usize).from_slice(device, edge_values, &.{ 2, edge_values.len / 2 }, .{
        .graph = graph,
    });

    const output = try model.forward(input, edge_index);
    const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, labels);
    graph.teardown(&loss.node);
}

fn serializeAffineCheckpoint(
    allocator: std.mem.Allocator,
    weights: anytype,
    biases: anytype,
) ![]u8 {
    var params = zg.LayerMap.init(allocator);
    defer params.deinit();

    for (weights, biases, 0..) |weight, bias, index| {
        var weight_name_buf: [32]u8 = undefined;
        const weight_name = try std.fmt.bufPrint(&weight_name_buf, "weights.{d}", .{index});
        try params.put(weight_name, weight, .{});

        var bias_name_buf: [32]u8 = undefined;
        const bias_name = try std.fmt.bufPrint(&bias_name_buf, "biases.{d}", .{index});
        try params.put(bias_name, bias, .{});
    }

    return params.serialize(allocator);
}

fn serializeSatoshiLmCheckpoint(
    allocator: std.mem.Allocator,
    model: *SatoshiLmModel(f32),
) ![]u8 {
    var params = zg.LayerMap.init(allocator);
    defer params.deinit();

    try params.put("token_embedding", model.token_embedding, .{});
    try params.put("position_embedding", model.position_embedding, .{});
    try params.put("query_weights", model.query_weights, .{});
    try params.put("query_bias", model.query_bias, .{});
    try params.put("key_weights", model.key_weights, .{});
    try params.put("key_bias", model.key_bias, .{});
    try params.put("value_weights", model.value_weights, .{});
    try params.put("value_bias", model.value_bias, .{});
    try params.put("output_weights", model.output_weights, .{});
    try params.put("output_bias", model.output_bias, .{});

    return params.serialize(allocator);
}

fn oneAffineCheckpointImport(
    comptime Model: type,
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    checkpoint: []const u8,
) !void {
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var params = try zg.LayerMap.deserialize(checkpoint, allocator, device, .{
        .requires_grad = true,
        .acquired = true,
        .owning = false,
        .graph = &graph,
    });
    defer params.deinit();

    var model = try params.extract(Model, "", .{});
    defer model.deinit();
}

fn serializeGcnCheckpoint(
    allocator: std.mem.Allocator,
    model: *GcnBenchmarkModel(f32),
) ![]u8 {
    var params = zg.LayerMap.init(allocator);
    defer params.deinit();

    try params.put(model.conv1.weights.get_label().?, model.conv1.weights, .{});
    try params.put(model.conv1.bias.get_label().?, model.conv1.bias, .{});
    try params.put(model.conv2.weights.get_label().?, model.conv2.weights, .{});
    try params.put(model.conv2.bias.get_label().?, model.conv2.bias, .{});

    return params.serialize(allocator);
}

fn oneSatoshiLmCheckpointImport(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    checkpoint: []const u8,
    context_len: usize,
    vocab_size: usize,
    hidden_size: usize,
) !void {
    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var params = try zg.LayerMap.deserialize(checkpoint, allocator, device, .{
        .requires_grad = true,
        .acquired = true,
        .owning = false,
        .graph = &graph,
    });
    defer params.deinit();

    const pack = try params.extract(SatoshiLmModel(f32).ParameterPack, "", .{});
    var model = try SatoshiLmModel(f32).fromParameterPack(
        device,
        context_len,
        vocab_size,
        hidden_size,
        pack,
        &graph,
    );
    defer model.deinit();
}

fn onePendulumCheckpointImport(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    checkpoint: []const u8,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const ParameterPack = struct {
        weights: [3]*Tensor,
        biases: [3]*Tensor,
    };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var params = try zg.LayerMap.deserialize(checkpoint, allocator, device, .{
        .requires_grad = true,
        .acquired = true,
        .owning = false,
        .graph = &graph,
    });
    defer params.deinit();

    const pack = try params.extract(ParameterPack, "", .{});
    var model: PendulumDynamicsModel(f32) = .{
        .weights = pack.weights,
        .biases = pack.biases,
        .input_size = input_size,
        .hidden_size = hidden_size,
        .output_size = output_size,
    };
    defer model.deinit();
}

fn oneCorridorCheckpointImport(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    checkpoint: []const u8,
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
) !void {
    const Tensor = zg.NDTensor(f32);
    const ParameterPack = struct {
        weights: [3]*Tensor,
        biases: [3]*Tensor,
    };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var params = try zg.LayerMap.deserialize(checkpoint, allocator, device, .{
        .requires_grad = true,
        .acquired = true,
        .owning = false,
        .graph = &graph,
    });
    defer params.deinit();

    const pack = try params.extract(ParameterPack, "", .{});
    var model: CorridorControlModel(f32) = .{
        .weights = pack.weights,
        .biases = pack.biases,
        .input_size = input_size,
        .hidden_size = hidden_size,
        .output_size = output_size,
    };
    defer model.deinit();
}

fn oneGcnCheckpointImport(
    allocator: std.mem.Allocator,
    device: zg.DeviceReference,
    checkpoint: []const u8,
) !void {
    const Tensor = zg.NDTensor(f32);
    const ParameterPack = struct {
        bench: struct {
            gcn: struct {
                conv1: struct {
                    weights: *Tensor,
                    bias: *Tensor,
                },
                conv2: struct {
                    weights: *Tensor,
                    bias: *Tensor,
                },
            },
        },
    };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var params = try zg.LayerMap.deserialize(checkpoint, allocator, device, .{
        .requires_grad = true,
        .acquired = true,
        .owning = false,
        .graph = &graph,
    });
    defer params.deinit();

    const pack = try params.extract(ParameterPack, "", .{});
    var model: GcnBenchmarkModel(f32) = .{
        .conv1 = .{
            .device = device,
            .graph = &graph,
            .weights = pack.bench.gcn.conv1.weights,
            .bias = pack.bench.gcn.conv1.bias,
        },
        .conv2 = .{
            .device = device,
            .graph = &graph,
            .weights = pack.bench.gcn.conv2.weights,
            .bias = pack.bench.gcn.conv2.bias,
        },
    };
    defer model.deinit();
}

fn interopCheckpointOutput(
    shapes: []const result.ShapeMetadata,
    setup_latency_ns: u64,
    timings_ns: []const u64,
    artifact_bytes: usize,
    host_blas_telemetry: ?result.HostBlasTelemetry,
    notes: ?[]const u8,
) RunOutput {
    return .{
        .shapes = shapes,
        .batch_size = null,
        .setup_latency_ns = setup_latency_ns,
        .timings_ns = timings_ns,
        .throughput_items = artifact_bytes,
        .throughput_unit = "bytes",
        .interop = .{
            .artifact_bytes = artifact_bytes,
            .tensor_count = shapes.len,
        },
        .host_blas_telemetry = host_blas_telemetry,
        .notes = notes,
    };
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

fn makeActionMask(
    allocator: std.mem.Allocator,
    action_values: []const usize,
    action_count: usize,
) ![]f32 {
    const values = try allocator.alloc(f32, action_values.len * action_count);
    @memset(values, 0);
    for (action_values, 0..) |action, row| {
        values[(row * action_count) + action] = 1.0;
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

fn makeSatoshiLmTokenStream(
    allocator: std.mem.Allocator,
    batch_size: usize,
    context_len: usize,
    vocab_size: usize,
    seed: u64,
) ![]usize {
    const values = try allocator.alloc(usize, batch_size + context_len);
    for (values, 0..) |*value, index| {
        value.* = @as(usize, @intCast(splitmix64(seed +% @as(u64, index)) % vocab_size));
    }
    return values;
}

fn makeSatoshiLmInputBatch(
    allocator: std.mem.Allocator,
    token_stream: []const usize,
    batch_size: usize,
    context_len: usize,
    vocab_size: usize,
) ![]f32 {
    const values = try allocator.alloc(f32, batch_size * context_len * vocab_size);
    @memset(values, 0);

    for (0..batch_size) |row| {
        for (0..context_len) |column| {
            const token = token_stream[row + column];
            values[((row * context_len + column) * vocab_size) + token] = 1;
        }
    }

    return values;
}

fn makeSatoshiLmLabelBatch(
    allocator: std.mem.Allocator,
    token_stream: []const usize,
    batch_size: usize,
    context_len: usize,
    vocab_size: usize,
) ![]f32 {
    const values = try allocator.alloc(f32, batch_size * vocab_size);
    @memset(values, 0);

    for (0..batch_size) |row| {
        const token = token_stream[row + context_len];
        values[(row * vocab_size) + token] = 1;
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

fn shapeMetadataFromSatoshiLm(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const label_shape_count: usize = if (spec.label_shape == null) 0 else 1;
    const shapes = try allocator.alloc(result.ShapeMetadata, 1 + label_shape_count);
    shapes[0] = .{ .name = "input", .dims = spec.input_shape.? };
    if (spec.label_shape) |label_shape| {
        shapes[1] = .{ .name = "labels", .dims = label_shape };
    }
    return shapes;
}

fn shapeMetadataFromPendulum(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const label_shape_count: usize = if (spec.label_shape == null) 0 else 1;
    const shapes = try allocator.alloc(result.ShapeMetadata, 1 + label_shape_count);
    shapes[0] = .{ .name = "input", .dims = spec.input_shape.? };
    if (spec.label_shape) |label_shape| {
        shapes[1] = .{ .name = "labels", .dims = label_shape };
    }
    return shapes;
}

fn shapeMetadataFromCorridorTrain(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const batch_size = spec.batch_size.?;
    const shapes = try allocator.alloc(result.ShapeMetadata, 5);
    shapes[0] = .{ .name = "state", .dims = spec.input_shape.? };
    shapes[1] = .{ .name = "next_state", .dims = spec.input_shape.? };
    shapes[2] = .{ .name = "action", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    shapes[3] = .{ .name = "reward", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    shapes[4] = .{ .name = "done", .dims = try allocDims(allocator, &.{ batch_size, 1 }) };
    return shapes;
}

fn shapeMetadataFromCorridorInfer(allocator: std.mem.Allocator, spec: manifest.Spec) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 1);
    shapes[0] = .{ .name = "state", .dims = spec.input_shape.? };
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

fn corridorHiddenSize() usize {
    return 32;
}

fn shapeMetadataFromMnistCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 6);
    shapes[0] = .{ .name = "weights.0", .dims = try allocDims(allocator, &.{ 128, 784 }) };
    shapes[1] = .{ .name = "biases.0", .dims = try allocDims(allocator, &.{128}) };
    shapes[2] = .{ .name = "weights.1", .dims = try allocDims(allocator, &.{ 64, 128 }) };
    shapes[3] = .{ .name = "biases.1", .dims = try allocDims(allocator, &.{64}) };
    shapes[4] = .{ .name = "weights.2", .dims = try allocDims(allocator, &.{ 10, 64 }) };
    shapes[5] = .{ .name = "biases.2", .dims = try allocDims(allocator, &.{10}) };
    return shapes;
}

fn shapeMetadataFromDqnCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 6);
    shapes[0] = .{ .name = "weights.0", .dims = try allocDims(allocator, &.{ 128, 4 }) };
    shapes[1] = .{ .name = "biases.0", .dims = try allocDims(allocator, &.{128}) };
    shapes[2] = .{ .name = "weights.1", .dims = try allocDims(allocator, &.{ 128, 128 }) };
    shapes[3] = .{ .name = "biases.1", .dims = try allocDims(allocator, &.{128}) };
    shapes[4] = .{ .name = "weights.2", .dims = try allocDims(allocator, &.{ 2, 128 }) };
    shapes[5] = .{ .name = "biases.2", .dims = try allocDims(allocator, &.{2}) };
    return shapes;
}

fn shapeMetadataFromSatoshiLmCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const vocab_size = interopSatoshiLmVocabSize();
    const hidden_size = charLmHiddenSize(vocab_size);
    const context_len = interopSatoshiLmContextLen();
    const shapes = try allocator.alloc(result.ShapeMetadata, 10);
    shapes[0] = .{ .name = "token_embedding", .dims = try allocDims(allocator, &.{ hidden_size, vocab_size }) };
    shapes[1] = .{ .name = "position_embedding", .dims = try allocDims(allocator, &.{ context_len, hidden_size }) };
    shapes[2] = .{ .name = "query_weights", .dims = try allocDims(allocator, &.{ hidden_size, hidden_size }) };
    shapes[3] = .{ .name = "query_bias", .dims = try allocDims(allocator, &.{hidden_size}) };
    shapes[4] = .{ .name = "key_weights", .dims = try allocDims(allocator, &.{ hidden_size, hidden_size }) };
    shapes[5] = .{ .name = "key_bias", .dims = try allocDims(allocator, &.{hidden_size}) };
    shapes[6] = .{ .name = "value_weights", .dims = try allocDims(allocator, &.{ hidden_size, hidden_size }) };
    shapes[7] = .{ .name = "value_bias", .dims = try allocDims(allocator, &.{hidden_size}) };
    shapes[8] = .{ .name = "output_weights", .dims = try allocDims(allocator, &.{ vocab_size, hidden_size }) };
    shapes[9] = .{ .name = "output_bias", .dims = try allocDims(allocator, &.{vocab_size}) };
    return shapes;
}

fn shapeMetadataFromPendulumCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 6);
    shapes[0] = .{ .name = "weights.0", .dims = try allocDims(allocator, &.{ pendulumHiddenSize(), pendulum_data.input_feature_count }) };
    shapes[1] = .{ .name = "biases.0", .dims = try allocDims(allocator, &.{pendulumHiddenSize()}) };
    shapes[2] = .{ .name = "weights.1", .dims = try allocDims(allocator, &.{ pendulumHiddenSize(), pendulumHiddenSize() }) };
    shapes[3] = .{ .name = "biases.1", .dims = try allocDims(allocator, &.{pendulumHiddenSize()}) };
    shapes[4] = .{ .name = "weights.2", .dims = try allocDims(allocator, &.{ pendulum_data.output_feature_count, pendulumHiddenSize() }) };
    shapes[5] = .{ .name = "biases.2", .dims = try allocDims(allocator, &.{pendulum_data.output_feature_count}) };
    return shapes;
}

fn shapeMetadataFromCorridorCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 6);
    shapes[0] = .{ .name = "weights.0", .dims = try allocDims(allocator, &.{ corridorHiddenSize(), corridor.state_feature_count }) };
    shapes[1] = .{ .name = "biases.0", .dims = try allocDims(allocator, &.{corridorHiddenSize()}) };
    shapes[2] = .{ .name = "weights.1", .dims = try allocDims(allocator, &.{ corridorHiddenSize(), corridorHiddenSize() }) };
    shapes[3] = .{ .name = "biases.1", .dims = try allocDims(allocator, &.{corridorHiddenSize()}) };
    shapes[4] = .{ .name = "weights.2", .dims = try allocDims(allocator, &.{ corridor.action_count, corridorHiddenSize() }) };
    shapes[5] = .{ .name = "biases.2", .dims = try allocDims(allocator, &.{corridor.action_count}) };
    return shapes;
}

fn shapeMetadataFromGcnCheckpoint(allocator: std.mem.Allocator) ![]const result.ShapeMetadata {
    const shapes = try allocator.alloc(result.ShapeMetadata, 4);
    shapes[0] = .{ .name = "bench.gcn.conv1.weights", .dims = try allocDims(allocator, &.{ 16, interopGcnInputFeatureCount() }) };
    shapes[1] = .{ .name = "bench.gcn.conv1.bias", .dims = try allocDims(allocator, &.{16}) };
    shapes[2] = .{ .name = "bench.gcn.conv2.weights", .dims = try allocDims(allocator, &.{ interopGcnOutputFeatureCount(), 16 }) };
    shapes[3] = .{ .name = "bench.gcn.conv2.bias", .dims = try allocDims(allocator, &.{interopGcnOutputFeatureCount()}) };
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

fn compilerCaptureMemoryStats(
    host: ?*const zg.device.HostDevice,
    peak_graph_arena_bytes: usize,
    final_graph_arena_bytes: usize,
) result.MemoryStats {
    var stats = result.MemoryStats{
        .peak_graph_arena_bytes = @as(u64, @intCast(peak_graph_arena_bytes)),
        .final_graph_arena_bytes = @as(u64, @intCast(final_graph_arena_bytes)),
    };

    if (host) |value| {
        const telemetry = value.cacheTelemetry();
        stats.peak_live_bytes = @as(u64, @intCast(telemetry.peak_live_bytes));
        stats.final_live_bytes = @as(u64, @intCast(telemetry.live_bytes));
        stats.peak_scratch_bytes = @as(u64, @intCast(telemetry.peak_scratch_bytes));
    }

    return stats;
}

fn charLmHiddenSize(vocab_size: usize) usize {
    return @max(vocab_size * 2, 64);
}

fn interopSatoshiLmContextLen() usize {
    return 16;
}

fn interopSatoshiLmVocabSize() usize {
    return 32;
}

fn interopGcnInputFeatureCount() usize {
    return 64;
}

fn interopGcnOutputFeatureCount() usize {
    return 7;
}

fn pendulumHiddenSize() usize {
    return 48;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_live_bytes.? >= output.memory.?.final_live_bytes.?);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler mnist capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 4, 1, 28, 28 };
    const label_shape = [_]usize{ 4, 10 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.mnist.capture",
        .suite = .compiler,
        .kind = .compiler_mnist_mlp_capture,
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
            "capture forward plus loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler satoshi lm capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 12, 24 };
    const label_shape = [_]usize{ 8, 24 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.satoshi-lm.capture",
        .suite = .compiler,
        .kind = .compiler_satoshi_lm_capture,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .label_shape = label_shape[0..],
        .provenance = inlineProvenance(&.{
            "derive a deterministic token stream",
            "materialize one-hot causal context windows",
            "capture forward plus loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler pendulum capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 4 };
    const label_shape = [_]usize{ 8, 3 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.pendulum.capture",
        .suite = .compiler,
        .kind = .compiler_pendulum_dynamics_capture,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .label_shape = label_shape[0..],
        .provenance = inlineProvenance(&.{
            "generate deterministic pendulum states",
            "simulate next-state regression targets",
            "capture forward plus regression loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler corridor capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 3 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.corridor.capture",
        .suite = .compiler,
        .kind = .compiler_corridor_control_capture,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .provenance = inlineProvenance(&.{
            "sample deterministic corridor start states",
            "simulate one-step corridor transitions",
            "capture Q-learning loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 5), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler dqn capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 4 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.dqn.capture",
        .suite = .compiler,
        .kind = .compiler_dqn_cartpole_capture,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .provenance = inlineProvenance(&.{
            "reshape states to input_shape",
            "derive deterministic transitions and targets",
            "capture Q-learning loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 5), output.shapes.len);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run compiler gcn capture benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 16, 8 };
    const label_shape = [_]usize{ 16, 7 };
    const spec: manifest.Spec = .{
        .id = "test.compiler.gcn.capture",
        .suite = .compiler,
        .kind = .compiler_gcn_capture,
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
            "capture forward plus loss graph without backward execution",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 3), output.shapes.len);
    try std.testing.expect(output.batch_size == null);
    try std.testing.expect(output.memory != null);
    try std.testing.expect(output.memory.?.peak_graph_arena_bytes.? > 0);
}

test "run interop satoshi lm export benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const spec: manifest.Spec = .{
        .id = "test.interop.satoshi-lm.export",
        .suite = .interop,
        .kind = .interop_satoshi_lm_safetensors_export,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .seed = 1,
        .provenance = inlineProvenance(&.{
            "materialize deterministic benchmark satoshi-lm parameters",
            "encode affine parameter stack as safetensors bytes",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 10), output.shapes.len);
    try std.testing.expectEqualStrings("token_embedding", output.shapes[0].name);
    try std.testing.expectEqualStrings("bytes", output.throughput_unit.?);
    try std.testing.expectEqual(output.throughput_items.?, output.interop.?.artifact_bytes);
    try std.testing.expectEqual(output.shapes.len, output.interop.?.tensor_count);
}

test "run interop gcn import benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const spec: manifest.Spec = .{
        .id = "test.interop.gcn.import",
        .suite = .interop,
        .kind = .interop_gcn_safetensors_import,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .seed = 1,
        .provenance = inlineProvenance(&.{
            "materialize deterministic benchmark gcn parameters",
            "encode checkpoint fixture as safetensors bytes",
            "decode graph-conv parameter stack from safetensors bytes",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 4), output.shapes.len);
    try std.testing.expectEqualStrings("bench.gcn.conv1.weights", output.shapes[0].name);
    try std.testing.expectEqualStrings("bytes", output.throughput_unit.?);
    try std.testing.expectEqual(output.throughput_items.?, output.interop.?.artifact_bytes);
    try std.testing.expectEqual(output.shapes.len, output.interop.?.tensor_count);
}

test "run satoshi lm infer benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 8, 12, 24 };
    const spec: manifest.Spec = .{
        .id = "test.satoshi-lm.infer",
        .suite = .model_infer,
        .kind = .satoshi_lm_infer,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 8,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .provenance = inlineProvenance(&.{
            "derive a deterministic token stream",
            "materialize one-hot causal context windows",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 1), output.shapes.len);
    try std.testing.expectEqualStrings("input", output.shapes[0].name);
    try std.testing.expectEqual(@as(usize, 8), output.batch_size.?);
}

test "run pendulum train benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 16, 4 };
    const label_shape = [_]usize{ 16, 3 };
    const spec: manifest.Spec = .{
        .id = "test.pendulum.train",
        .suite = .model_train,
        .kind = .pendulum_dynamics_train,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 16,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .label_shape = label_shape[0..],
        .provenance = inlineProvenance(&.{
            "generate deterministic pendulum states",
            "encode theta as sine/cosine features",
            "simulate next-state regression targets",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 2), output.shapes.len);
    try std.testing.expectEqualStrings("labels", output.shapes[1].name);
}

test "run pendulum infer benchmark" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const input_shape = [_]usize{ 16, 4 };
    const spec: manifest.Spec = .{
        .id = "test.pendulum.infer",
        .suite = .model_infer,
        .kind = .pendulum_dynamics_infer,
        .dtype = .f32,
        .warmup_iterations = 0,
        .measured_iterations = 1,
        .batch_size = 16,
        .thread_count = 1,
        .seed = 1,
        .input_shape = input_shape[0..],
        .provenance = inlineProvenance(&.{
            "generate deterministic pendulum states",
            "encode theta as sine/cosine features",
        }),
        .path = "inline",
    };

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 1), output.shapes.len);
    try std.testing.expectEqualStrings("input", output.shapes[0].name);
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.ok, run_result.status);
    const output = run_result.output orelse return error.MissingBenchmarkRunOutput;
    try std.testing.expectEqual(@as(usize, 1), output.timings_ns.len);
    try std.testing.expectEqual(@as(usize, 3), output.shapes.len);
    try std.testing.expectEqualStrings("edge_index", output.shapes[1].name);
}

test "memory suite skips cuda requests explicitly" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const buffer_shape = [_]usize{ 8, 8 };
    const spec: manifest.Spec = .{
        .id = "test.memory.tensor-cache.cuda",
        .suite = .memory,
        .kind = .memory_tensor_cache_cycle,
        .device = .{ .kind = .cuda, .cuda_device_index = 0 },
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

    const run_result = try run(arena.allocator(), spec);
    try std.testing.expectEqual(result.Status.skipped, run_result.status);
    try std.testing.expect(run_result.output == null);
    try std.testing.expect(run_result.notes != null);
    try std.testing.expectEqualStrings("cuda", run_result.backend.device);
}
