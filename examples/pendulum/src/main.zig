const std = @import("std");
const zg = @import("zigrad");

const pendulum_data = @import("dataset.zig");
const DynamicsConfig = pendulum_data.DynamicsConfig;
const PendulumDataset = pendulum_data.PendulumDataset;
const PendulumDynamicsModel = @import("model.zig").PendulumDynamicsModel;

const std_options = .{ .log_level = .info };
const T = f32;

pub const RunConfig = struct {
    batch_size: usize = 32,
    num_epochs: usize = 24,
    train_samples: usize = 512,
    eval_samples: usize = 160,
    hidden_size: usize = 48,
    learning_rate: T = 0.01,
    train_seed: u64 = 20260328,
    eval_seed: u64 = 20260329,
    rollout_seed: u64 = 20260330,
    rollout_steps: usize = 32,
    verbose: ?bool = null,
    load_path: ?[]const u8 = "pendulum.stz",
    save_path: ?[]const u8 = "pendulum.stz",
    device_request: ?zg.device.RuntimeDeviceRequest = null,
    dynamics: DynamicsConfig = .{},
};

pub const RunSummary = struct {
    epochs_completed: usize,
    train_batches: usize,
    initial_loss: T,
    final_loss: T,
    rollout_rmse: T,
};

pub fn runPendulum() !void {
    _ = try runPendulumWithConfig(.{});
}

pub fn trainPendulumSmoke() !RunSummary {
    return runPendulumWithConfig(.{
        .batch_size = 16,
        .num_epochs = 18,
        .train_samples = 256,
        .eval_samples = 96,
        .hidden_size = 48,
        .learning_rate = 0.012,
        .train_seed = 7,
        .eval_seed = 11,
        .rollout_seed = 13,
        .rollout_steps = 24,
        .load_path = null,
        .save_path = null,
        .verbose = false,
    });
}

pub fn runPendulumWithConfig(config: RunConfig) !RunSummary {
    if (config.batch_size == 0) return error.InvalidBatchSize;
    if (config.num_epochs == 0) return error.InvalidEpochCount;
    if (config.train_samples == 0 or config.eval_samples == 0) return error.InvalidSampleCount;
    if (config.hidden_size == 0) return error.InvalidHiddenSize;
    if (config.rollout_steps == 0) return error.InvalidRolloutLength;

    const allocator = std.heap.smp_allocator;

    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    var runtime_device = try zg.device.initRuntimeDevice(config.device_request, .{ .allow_cuda = true });
    defer runtime_device.deinit();
    _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "pendulum:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "pendulum:summary",
        .include_telemetry = true,
    }) catch {};
    const device = runtime_device.reference();

    var train_dataset = try PendulumDataset.initGenerated(
        allocator,
        config.dynamics,
        config.train_samples,
        config.batch_size,
        config.train_seed,
    );
    defer train_dataset.deinit();

    var eval_dataset = try PendulumDataset.initGenerated(
        allocator,
        config.dynamics,
        config.eval_samples,
        config.batch_size,
        config.eval_seed,
    );
    defer eval_dataset.deinit();

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = config.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();

    var model = if (config.load_path) |load_path|
        PendulumDynamicsModel(T).load(
            load_path,
            device,
            pendulum_data.input_feature_count,
            config.hidden_size,
            pendulum_data.output_feature_count,
        ) catch |err| switch (err) {
            std.fs.File.OpenError.FileNotFound => try PendulumDynamicsModel(T).initWithGraphAndSeed(
                device,
                pendulum_data.input_feature_count,
                config.hidden_size,
                pendulum_data.output_feature_count,
                null,
                config.train_seed,
            ),
            else => return err,
        }
    else
        try PendulumDynamicsModel(T).initWithGraphAndSeed(
            device,
            pendulum_data.input_feature_count,
            config.hidden_size,
            pendulum_data.output_feature_count,
            null,
            config.train_seed,
        );
    defer model.deinit();
    try model.attach_optimizer(optimizer);

    const verbose = config.verbose orelse ((std.posix.getenv("ZG_VERBOSE") orelse "0")[0] == '1');
    const initial_loss = try evaluateAverageLoss(&model, &eval_dataset, device);

    var train_batches: usize = 0;
    for (0..config.num_epochs) |epoch_index| {
        const epoch_result = try trainOneEpoch(&model, &train_dataset, device, optimizer);
        train_batches += epoch_result.batches;
        if (verbose) {
            std.debug.print("epoch {d:>2}: avg_loss={d:.6}\n", .{
                epoch_index + 1,
                epoch_result.average_loss,
            });
        }
    }

    const final_loss = try evaluateAverageLoss(&model, &eval_dataset, device);
    const rollout_rmse = try evaluateRolloutRmse(
        allocator,
        &model,
        device,
        config.dynamics,
        config.rollout_seed,
        config.rollout_steps,
    );

    std.debug.print(
        "pendulum: initial_loss={d:.6}, final_loss={d:.6}, rollout_rmse={d:.6}, batches={d}\n",
        .{ initial_loss, final_loss, rollout_rmse, train_batches },
    );

    if (config.save_path) |save_path| {
        try model.save(save_path);
    }

    return .{
        .epochs_completed = config.num_epochs,
        .train_batches = train_batches,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .rollout_rmse = rollout_rmse,
    };
}

pub fn main() !void {
    const smoke = (std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1';

    _ = try runPendulumWithConfig(if (smoke)
        .{
            .batch_size = 16,
            .num_epochs = 18,
            .train_samples = 256,
            .eval_samples = 96,
            .hidden_size = 48,
            .learning_rate = 0.012,
            .train_seed = 7,
            .eval_seed = 11,
            .rollout_seed = 13,
            .rollout_steps = 24,
            .load_path = null,
            .save_path = null,
            .verbose = false,
        }
    else
        .{});
}

fn trainOneEpoch(
    model: *PendulumDynamicsModel(T),
    dataset: *const PendulumDataset,
    device: zg.DeviceReference,
    optimizer: zg.Optimizer,
) !struct { average_loss: T, batches: usize } {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = true;
    defer zg.runtime.grad_enabled = previous_grad_state;

    var loss_sum: T = 0;
    const batch_count = dataset.batchCount();

    for (0..batch_count) |batch_index| {
        var batch = try dataset.makeBatch(device, batch_index, null);
        defer batch.deinit();

        const prediction = try model.forward(batch.inputs);
        defer prediction.deinit();

        const loss = try zg.loss.mse_loss(T, prediction, batch.targets);
        defer loss.deinit();

        loss_sum += loss.get(0);
        try loss.backward();
        try optimizer.step();
        model.zero_grad();
    }

    return .{
        .average_loss = loss_sum / @as(T, @floatFromInt(batch_count)),
        .batches = batch_count,
    };
}

fn evaluateAverageLoss(
    model: *PendulumDynamicsModel(T),
    dataset: *const PendulumDataset,
    device: zg.DeviceReference,
) !T {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    var loss_sum: T = 0;
    const batch_count = dataset.batchCount();

    for (0..batch_count) |batch_index| {
        var batch = try dataset.makeBatch(device, batch_index, null);
        defer batch.deinit();

        const prediction = try model.forward(batch.inputs);
        defer prediction.deinit();

        const loss = try zg.loss.mse_loss(T, prediction, batch.targets);
        defer loss.deinit();

        loss_sum += loss.get(0);
    }

    return loss_sum / @as(T, @floatFromInt(batch_count));
}

fn evaluateRolloutRmse(
    allocator: std.mem.Allocator,
    model: *PendulumDynamicsModel(T),
    device: zg.DeviceReference,
    dynamics: DynamicsConfig,
    rollout_seed: u64,
    rollout_steps: usize,
) !T {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const torques = try pendulum_data.makeRolloutTorques(allocator, dynamics, rollout_steps, rollout_seed);
    defer allocator.free(torques);

    var predicted_state = pendulum_data.deterministicInitialState(dynamics, rollout_seed +% 101);
    var reference_state = predicted_state;
    var squared_error_sum: f64 = 0;

    for (torques) |torque| {
        const input_values = dynamics.normalizeInput(predicted_state, torque);
        const input = try zg.NDTensor(T).from_slice(device, &input_values, &.{ 1, pendulum_data.input_feature_count }, .{});
        defer input.deinit();

        const prediction = try model.forward(input);
        defer prediction.deinit();

        const prediction_host = try prediction.to_host_owned(allocator);
        defer allocator.free(prediction_host);

        const predicted_next = dynamics.denormalizeState(.{
            prediction_host[0],
            prediction_host[1],
            prediction_host[2],
        });
        const reference_next = dynamics.simulateStep(reference_state, torque);

        const theta_error = pendulum_data.angleDifference(predicted_next[0], reference_next[0]);
        const omega_error = predicted_next[1] - reference_next[1];
        squared_error_sum += (@as(f64, @floatCast(theta_error * theta_error)) +
            @as(f64, @floatCast(omega_error * omega_error)));

        predicted_state = predicted_next;
        reference_state = reference_next;
    }

    return @as(T, @floatCast(std.math.sqrt(
        squared_error_sum / @as(f64, @floatFromInt(rollout_steps * pendulum_data.output_feature_count)),
    )));
}

test "run pendulum smoke" {
    const summary = try trainPendulumSmoke();
    try std.testing.expectEqual(@as(usize, 18), summary.epochs_completed);
    try std.testing.expect(summary.train_batches > 0);
    try std.testing.expect(std.math.isFinite(summary.initial_loss));
    try std.testing.expect(std.math.isFinite(summary.final_loss));
    try std.testing.expect(std.math.isFinite(summary.rollout_rmse));
    try std.testing.expect(summary.final_loss < summary.initial_loss);
    try std.testing.expect(summary.rollout_rmse < 0.35);
}
