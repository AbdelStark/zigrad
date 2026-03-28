const std = @import("std");
const zg = @import("zigrad");

const corridor = @import("environment.zig");
const CorridorEnv = corridor.CorridorEnv;
const CorridorConfig = corridor.CorridorConfig;
const CorridorControlModel = @import("model.zig").CorridorControlModel;
const ReplayBuffer = @import("replay_buffer.zig").ReplayBuffer;

const std_options = .{ .log_level = .info };
const T = corridor.T;

const Transition = struct {
    state: [corridor.state_feature_count]T,
    action: usize,
    next_state: [corridor.state_feature_count]T,
    reward: T,
    done: T,
};

const EvaluationSummary = struct {
    average_return: T,
    success_rate: T,
};

pub const TrainConfig = struct {
    batch_size: usize = 32,
    num_episodes: usize = 180,
    replay_capacity: usize = 1024,
    replay_warmup_steps: usize = 24,
    train_every: usize = 1,
    hidden_size: usize = 32,
    learning_rate: T = 0.004,
    gamma: T = 0.97,
    tau: T = 0.08,
    eps_start: T = 0.9,
    eps_end: T = 0.05,
    eps_decay: T = 120.0,
    eval_interval: usize = 20,
    success_threshold: ?T = 1.0,
    seed: u64 = 20260328,
    load_path: ?[]const u8 = "corridor.stz",
    save_path: ?[]const u8 = "corridor.stz",
    verbose: ?bool = null,
    device_request: ?zg.device.RuntimeDeviceRequest = null,
    environment: CorridorConfig = .{},
};

pub const TrainSummary = struct {
    episodes_run: usize,
    total_steps: usize,
    optimization_steps: usize,
    initial_eval_return: T,
    final_eval_return: T,
    final_success_rate: T,
};

pub fn trainCorridor() !void {
    _ = try trainCorridorWithConfig(.{});
}

pub fn trainCorridorSmoke() !TrainSummary {
    return trainCorridorWithConfig(.{
        .batch_size = 16,
        .num_episodes = 90,
        .replay_capacity = 256,
        .replay_warmup_steps = 12,
        .hidden_size = 24,
        .learning_rate = 0.006,
        .gamma = 0.97,
        .tau = 0.12,
        .eps_decay = 80.0,
        .eval_interval = 10,
        .success_threshold = 1.0,
        .seed = 7,
        .load_path = null,
        .save_path = null,
        .verbose = false,
    });
}

pub fn trainCorridorWithConfig(config: TrainConfig) !TrainSummary {
    try validateConfig(config);

    const allocator = std.heap.smp_allocator;
    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    var runtime_device = try zg.device.initRuntimeDevice(config.device_request, .{ .allow_cuda = true });
    defer runtime_device.deinit();
    _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "corridor:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "corridor:summary",
        .include_telemetry = true,
    }) catch {};
    const device = runtime_device.reference();

    var optimizer_state = zg.optim.Adam.init(allocator, .{
        .lr = config.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .grad_clip_enabled = true,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
    });
    defer optimizer_state.deinit();
    const optimizer = optimizer_state.optimizer();

    var model = if (config.load_path) |load_path|
        CorridorControlModel(T).load(
            load_path,
            device,
            corridor.state_feature_count,
            config.hidden_size,
            corridor.action_count,
        ) catch |err| switch (err) {
            std.fs.File.OpenError.FileNotFound => try CorridorControlModel(T).initWithGraphAndSeed(
                device,
                corridor.state_feature_count,
                config.hidden_size,
                corridor.action_count,
                null,
                config.seed,
            ),
            else => return err,
        }
    else
        try CorridorControlModel(T).initWithGraphAndSeed(
            device,
            corridor.state_feature_count,
            config.hidden_size,
            corridor.action_count,
            null,
            config.seed,
        );
    defer model.deinit();
    try model.attach_optimizer(optimizer);

    var target = try model.clone();
    defer target.deinit();
    target.set_requires_grad(false);

    var replay_buffer = try ReplayBuffer(Transition).init(allocator, config.replay_capacity);
    defer replay_buffer.deinit();

    var env = CorridorEnv.init(config.seed +% 3, config.environment);
    var prng = std.Random.DefaultPrng.init(config.seed +% 5);

    const verbose = config.verbose orelse ((std.posix.getenv("ZG_VERBOSE") orelse "0")[0] == '1');
    const initial_eval = try evaluatePolicy(&model, device, config.environment);

    var total_steps: usize = 0;
    var optimization_steps: usize = 0;
    var episodes_run: usize = config.num_episodes;
    var final_eval = initial_eval;

    for (0..config.num_episodes) |episode_index| {
        var state = env.reset();
        var episode_reward: T = 0;
        var loss_sum: T = 0;
        var loss_count: usize = 0;

        for (0..config.environment.max_steps) |_| {
            const epsilon = epsilonAtStep(config, total_steps);
            const action = try selectAction(&model, state, epsilon, &prng, device);
            const step_result = env.step(action);

            replay_buffer.add(.{
                .state = state,
                .action = @intFromEnum(action),
                .next_state = step_result.state,
                .reward = step_result.reward,
                .done = if (step_result.done) 1.0 else 0.0,
            });

            episode_reward += step_result.reward;
            state = step_result.state;
            total_steps += 1;

            const can_optimize = total_steps >= config.replay_warmup_steps and
                replay_buffer.size >= config.batch_size and
                total_steps % config.train_every == 0;
            if (can_optimize) {
                const loss = try trainOneBatch(
                    allocator,
                    &model,
                    &target,
                    &replay_buffer,
                    config.batch_size,
                    config.gamma,
                    config.tau,
                    optimizer,
                    prng.random(),
                    device,
                );
                optimization_steps += 1;
                loss_sum += loss;
                loss_count += 1;
            }

            if (step_result.done) break;
        }

        const should_evaluate = ((episode_index + 1) % config.eval_interval == 0) or
            (episode_index + 1 == config.num_episodes);
        if (should_evaluate) {
            final_eval = try evaluatePolicy(&model, device, config.environment);
            if (verbose) {
                const avg_loss = if (loss_count > 0)
                    loss_sum / @as(T, @floatFromInt(loss_count))
                else
                    0.0;
                std.debug.print(
                    "corridor episode {d:>3}: reward={d:.3}, avg_loss={d:.5}, eval_return={d:.3}, success={d:.2}%\n",
                    .{
                        episode_index + 1,
                        episode_reward,
                        avg_loss,
                        final_eval.average_return,
                        final_eval.success_rate * 100.0,
                    },
                );
            }

            if (config.success_threshold) |threshold| {
                if (final_eval.success_rate >= threshold) {
                    episodes_run = episode_index + 1;
                    break;
                }
            }
        }
    }

    if (config.save_path) |save_path| {
        try model.save(save_path);
    }

    std.debug.print(
        "corridor: initial_eval_return={d:.3}, final_eval_return={d:.3}, success_rate={d:.2}%, steps={d}, updates={d}\n",
        .{
            initial_eval.average_return,
            final_eval.average_return,
            final_eval.success_rate * 100.0,
            total_steps,
            optimization_steps,
        },
    );

    return .{
        .episodes_run = episodes_run,
        .total_steps = total_steps,
        .optimization_steps = optimization_steps,
        .initial_eval_return = initial_eval.average_return,
        .final_eval_return = final_eval.average_return,
        .final_success_rate = final_eval.success_rate,
    };
}

pub fn main() !void {
    const smoke = (std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1';
    _ = try trainCorridorWithConfig(if (smoke)
        .{
            .batch_size = 16,
            .num_episodes = 90,
            .replay_capacity = 256,
            .replay_warmup_steps = 12,
            .hidden_size = 24,
            .learning_rate = 0.006,
            .gamma = 0.97,
            .tau = 0.12,
            .eps_decay = 80.0,
            .eval_interval = 10,
            .success_threshold = 1.0,
            .seed = 7,
            .load_path = null,
            .save_path = null,
            .verbose = false,
        }
    else
        .{});
}

fn validateConfig(config: TrainConfig) !void {
    if (config.batch_size == 0) return error.InvalidBatchSize;
    if (config.num_episodes == 0) return error.InvalidEpisodeCount;
    if (config.replay_capacity == 0) return error.InvalidReplayCapacity;
    if (config.replay_warmup_steps >= config.replay_capacity) return error.InvalidReplayWarmup;
    if (config.hidden_size == 0) return error.InvalidHiddenSize;
    if (config.learning_rate <= 0) return error.InvalidLearningRate;
    if (config.train_every == 0) return error.InvalidTrainInterval;
    if (config.eval_interval == 0) return error.InvalidEvalInterval;
    if (config.eps_start < config.eps_end or config.eps_end < 0) return error.InvalidEpsilonSchedule;
    if (config.gamma <= 0 or config.gamma > 1.0) return error.InvalidDiscount;
    if (config.tau <= 0 or config.tau > 1.0) return error.InvalidTau;
    try config.environment.validate();
}

fn epsilonAtStep(config: TrainConfig, step: usize) T {
    return config.eps_end +
        ((config.eps_start - config.eps_end) * @exp(-@as(T, @floatFromInt(step)) / config.eps_decay));
}

fn selectAction(
    model: *CorridorControlModel(T),
    state: [corridor.state_feature_count]T,
    epsilon: T,
    prng: *std.Random.DefaultPrng,
    device: zg.DeviceReference,
) !corridor.Action {
    var random = prng.random();
    if (random.float(T) < epsilon) {
        return corridor.Action.fromIndex(random.uintLessThan(usize, corridor.action_count));
    }
    return greedyAction(model, state, device);
}

fn greedyAction(
    model: *CorridorControlModel(T),
    state: [corridor.state_feature_count]T,
    device: zg.DeviceReference,
) !corridor.Action {
    const Tensor = zg.NDTensor(T);
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const state_tensor = try Tensor.from_slice(device, state[0..], &.{ 1, corridor.state_feature_count }, .{});
    defer state_tensor.deinit();

    const q_values = try model.forward(state_tensor);
    defer q_values.deinit();

    var best_index: usize = 0;
    var best_value = q_values.get(0);
    for (1..corridor.action_count) |index| {
        const value = q_values.get(index);
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }

    return corridor.Action.fromIndex(best_index);
}

fn evaluatePolicy(
    model: *CorridorControlModel(T),
    device: zg.DeviceReference,
    environment: CorridorConfig,
) !EvaluationSummary {
    var env = CorridorEnv.init(0, environment);
    const scenario_count = @as(usize, @intCast(environment.goal_position - environment.pit_position - 1));

    var total_return: T = 0;
    var successes: usize = 0;

    for (0..scenario_count) |index| {
        const start = corridor.StartState{
            .position = environment.pit_position + 1 + @as(i32, @intCast(index)),
            .velocity = 0,
        };
        var state = try env.resetTo(start);
        var episode_return: T = 0;
        var reached_goal = false;

        for (0..environment.max_steps) |_| {
            const action = try greedyAction(model, state, device);
            const step_result = env.step(action);
            episode_return += step_result.reward;
            state = step_result.state;
            reached_goal = reached_goal or step_result.reached_goal;
            if (step_result.done) break;
        }

        total_return += episode_return;
        if (reached_goal) successes += 1;
    }

    return .{
        .average_return = total_return / @as(T, @floatFromInt(scenario_count)),
        .success_rate = @as(T, @floatFromInt(successes)) / @as(T, @floatFromInt(scenario_count)),
    };
}

fn trainOneBatch(
    allocator: std.mem.Allocator,
    model: *CorridorControlModel(T),
    target: *CorridorControlModel(T),
    replay_buffer: *const ReplayBuffer(Transition),
    batch_size: usize,
    gamma: T,
    tau: T,
    optimizer: zg.Optimizer,
    rng: std.Random,
    device: zg.DeviceReference,
) !T {
    const Tensor = zg.NDTensor(T);

    var batch = try replay_buffer.sampleWithoutReplacement(allocator, rng, batch_size);
    defer batch.deinit(allocator);

    const feature_count = corridor.state_feature_count;
    const state_values = try allocator.alloc(T, batch_size * feature_count);
    defer allocator.free(state_values);
    const next_state_values = try allocator.alloc(T, batch_size * feature_count);
    defer allocator.free(next_state_values);
    const gamma_values = try allocator.alloc(T, batch_size);
    defer allocator.free(gamma_values);
    const one_values = try allocator.alloc(T, batch_size);
    defer allocator.free(one_values);

    @memset(gamma_values, gamma);
    @memset(one_values, 1.0);

    for (batch.items(.state), 0..) |state, index| {
        @memcpy(state_values[index * feature_count ..][0..feature_count], state[0..]);
    }
    for (batch.items(.next_state), 0..) |next_state, index| {
        @memcpy(next_state_values[index * feature_count ..][0..feature_count], next_state[0..]);
    }

    const shape = &[_]usize{ batch_size, feature_count };
    const column_shape = &[_]usize{ batch_size, 1 };

    const states = try Tensor.from_slice(device, state_values, shape, .{});
    defer states.deinit();
    const next_states = try Tensor.from_slice(device, next_state_values, shape, .{});
    defer next_states.deinit();
    const actions = try zg.NDTensor(usize).from_slice(device, batch.items(.action), column_shape, .{});
    defer actions.deinit();
    const rewards = try Tensor.from_slice(device, batch.items(.reward), column_shape, .{});
    defer rewards.deinit();
    const dones = try Tensor.from_slice(device, batch.items(.done), column_shape, .{});
    defer dones.deinit();
    const gamma_tensor = try Tensor.from_slice(device, gamma_values, column_shape, .{});
    defer gamma_tensor.deinit();
    const ones = try Tensor.from_slice(device, one_values, column_shape, .{});
    defer ones.deinit();

    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const next_q_values = try target.forward(next_states);
    defer next_q_values.deinit();
    const max_next_q_values = try next_q_values.max_along(.{ .dim = 1, .keep_dims = true });
    defer max_next_q_values.deinit();

    const discounted = try max_next_q_values.mul(gamma_tensor);
    defer discounted.deinit();
    const not_done = try ones.sub(dones);
    defer not_done.deinit();
    const masked = try discounted.mul(not_done);
    defer masked.deinit();
    const targets = try rewards.add(masked);
    defer targets.deinit();

    zg.runtime.grad_enabled = true;

    const all_q_values = try model.forward(states);
    defer all_q_values.deinit();
    const selected_q_values = try all_q_values.gather(actions.data, 1);
    defer selected_q_values.deinit();

    const loss = try zg.loss.smooth_l1_loss(T, selected_q_values, targets, 1.0);
    defer loss.deinit();
    const loss_value = loss.get(0);

    try loss.backward();
    try optimizer.step();
    model.zero_grad();
    try target.soft_update_from(model, tau);

    return loss_value;
}
