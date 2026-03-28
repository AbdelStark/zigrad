///! NOTE: The underlying device abstractions have changed and this example has not yet been migrated
/// please disregard code surround allocators in this example until its migrated, everything else is fine.
///
/// NOTE: the replay buffer needs thought to optimize for device agnosticism + performance. I could write
/// it here, but this might hint at primitives we should provide the user.
const std = @import("std");
const zg = @import("zigrad");
const CartPole = @import("CartPole.zig");
const DQNAgent = @import("dqn_agent.zig").DQNAgent;
const tb = @import("tensorboard");
const T = f32;

pub const TrainConfig = struct {
    seed: ?u64 = null,
    tensorboard_log_dir: []const u8 = "/tmp/",
    max_steps: usize = 200,
    tau: T = 0.005,
    num_episodes: usize = 10_000,
    replay_warmup_steps: usize = 128,
    batch_size: usize = 128,
    train_every: usize = 1,
    early_stop_window: usize = 100,
    early_stop_reward: ?T = 195,
    device_request: ?zg.device.RuntimeDeviceRequest = null,
};

pub const TrainSummary = struct {
    episodes_run: usize,
    total_steps: usize,
    optimization_steps: usize,
    solved: bool,
};

pub fn trainDQN() !void {
    _ = try trainDQNWithConfig(.{});
}

pub fn trainDQNSmoke() !TrainSummary {
    return trainDQNWithConfig(.{
        .seed = 7,
        .tensorboard_log_dir = "/tmp/zigrad-dqn-smoke",
        .max_steps = 24,
        .tau = 0.01,
        .num_episodes = 6,
        .replay_warmup_steps = 8,
        .batch_size = 4,
        .train_every = 4,
        .early_stop_window = 0,
        .early_stop_reward = null,
    });
}

pub fn trainDQNWithConfig(config: TrainConfig) !TrainSummary {
    if (config.max_steps == 0) return error.InvalidMaxSteps;
    if (config.batch_size == 0) return error.InvalidBatchSize;
    if (config.train_every == 0) return error.InvalidTrainInterval;

    var runtime_device = try zg.device.initRuntimeDevice(config.device_request, .{ .allow_cuda = true });
    defer runtime_device.deinit();
    _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "dqn:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "dqn:summary",
        .include_telemetry = true,
    }) catch {};
    const device = runtime_device.reference();

    // Zigrad has a global graph that can be overriden for user-provided graphs.
    zg.global_graph_init(std.heap.smp_allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    // TODO: device side replay buffer, rm this allocator
    var im_pool = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer im_pool.deinit();
    const im_alloc = im_pool.allocator();

    // For tracking stats
    var arena = std.heap.ArenaAllocator.init(std.heap.raw_c_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Initialize environment and logger (old)
    const seed = config.seed orelse std.crypto.random.int(u64);
    var env = CartPole.init(seed);
    if (std.fs.path.isAbsolute(config.tensorboard_log_dir)) {
        std.fs.makeDirAbsolute(config.tensorboard_log_dir) catch |err| switch (err) {
            error.PathAlreadyExists => {},
            else => return err,
        };
    } else {
        try std.fs.cwd().makePath(config.tensorboard_log_dir);
    }
    var tb_logger = try tb.TensorBoardLogger.init(config.tensorboard_log_dir, allocator);
    defer tb_logger.deinit();

    // Configure optimizer with clipping
    var optimizer = zg.optim.Adam.init(allocator, .{
        .lr = 1e-4,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
        .grad_clip_enabled = true,
    });
    defer optimizer.deinit();

    var agent = try DQNAgent(T, 10_000, 3).init(device, .{
        .input_size = 4,
        .hidden_size = 128,
        .output_size = 2,
        .gamma = 0.99,
        .eps_start = 0.9,
        .eps_end = 0.05,
        .eps_decay = 1000,
    });
    defer agent.deinit();
    try agent.policy_net.attach_optimizer(optimizer.optimizer()); // register policy net params for training

    var total_rewards = try allocator.alloc(T, config.num_episodes);
    defer allocator.free(total_rewards);

    var total_steps: usize = 0;
    var optimization_steps: usize = 0;
    for (0..config.num_episodes) |episode| {
        var state: [4]T = env.reset();
        var total_reward: T = 0;
        var action_sum: T = 0;
        var loss_sum: T = 0;
        var loss_count: T = 0;
        var steps: usize = 0;

        // Training loop
        while (true) {
            const action = try agent.select_action(state, total_steps, device);
            action_sum += @as(T, @floatFromInt(action));
            const step_result = env.step(action);

            agent.store_transition(.{
                .state = state,
                .action = action,
                .next_state = step_result.state,
                .reward = step_result.reward,
                .done = @as(T, @floatFromInt(step_result.done)),
            });

            steps += 1;
            total_reward += step_result.reward;
            state = step_result.state;

            const can_train = total_steps >= config.replay_warmup_steps and
                agent.replay_buffer.size >= config.batch_size and
                total_steps % config.train_every == 0;
            if (can_train) {
                agent.policy_net.set_requires_grad(true);
                const loss = try agent.train(im_alloc, tb_logger, device, config.batch_size);
                try optimizer.optimizer().step();
                loss_sum += loss;
                loss_count += 1;
                optimization_steps += 1;
                try agent.update_target_network(config.tau);
                agent.policy_net.set_requires_grad(false);

                // Log training metrics to tensorboard
                try tb_logger.addScalar("training/loss", loss, @intCast(total_steps));
                try tb_logger.addScalar("training/epsilon", agent.eps, @intCast(total_steps));
            }

            total_steps += 1;
            _ = im_pool.reset(.retain_capacity); // (old)

            if (step_result.done > 0 or steps >= config.max_steps) break;
        }

        const avg_action: T = action_sum / @as(T, @floatFromInt(steps));
        total_rewards[episode] = total_reward;

        // Log episode metrics
        if (loss_count > 0) {
            const avg_loss = loss_sum / loss_count;
            try tb_logger.addScalar("episode/reward", total_reward, @intCast(episode));
            try tb_logger.addScalar("episode/avg_loss", avg_loss, @intCast(episode));
            try tb_logger.addScalar("episode/avg_action", avg_action, @intCast(episode));
        }

        // Track moving average and check for early stopping
        if (config.early_stop_reward) |early_stop_reward| {
            if (config.early_stop_window > 0 and episode + 1 >= config.early_stop_window) {
                const running_avg = blk: {
                    var sum: T = 0;
                    const start = (episode + 1) - config.early_stop_window;
                    for (total_rewards[start .. episode + 1]) |r| {
                        sum += r;
                    }
                    break :blk sum / @as(T, @floatFromInt(config.early_stop_window));
                };

                try tb_logger.addScalar("episode/running_avg", running_avg, @intCast(episode));

                if (running_avg >= early_stop_reward) {
                    std.debug.print("Solved in {d} episodes\n", .{episode + 1});
                    return .{
                        .episodes_run = episode + 1,
                        .total_steps = total_steps,
                        .optimization_steps = optimization_steps,
                        .solved = true,
                    };
                }
            }
        }
    }

    return .{
        .episodes_run = config.num_episodes,
        .total_steps = total_steps,
        .optimization_steps = optimization_steps,
        .solved = false,
    };
}
