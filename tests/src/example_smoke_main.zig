const std = @import("std");

const hello_world = @import("examples_hello_world_main");
const mnist_main = @import("examples_mnist_main");
const dqn_train = @import("examples_dqn_train");
const gcn_main = @import("examples_gcn_main");
const char_lm_main = @import("examples_char_lm_main");
const pendulum_main = @import("examples_pendulum_main");
const corridor_main = @import("examples_corridor_main");

const std_options = .{ .log_level = .info };

pub fn main() !void {
    try hello_world.main();

    const mnist_summary = try mnist_main.run_mnist_with_config(
        "examples/mnist/data/mnist_train_small.csv",
        "examples/mnist/data/mnist_test_small.csv",
        .{
            .batch_size = 16,
            .num_epochs = 1,
            .verbose = false,
            .load_path = null,
            .save_path = null,
        },
    );
    if (mnist_summary.epochs_completed != 1 or mnist_summary.train_batches == 0) {
        return error.MnistSmokeFailed;
    }

    const dqn_summary = try dqn_train.trainDQNSmoke();
    if (dqn_summary.episodes_run == 0 or
        dqn_summary.optimization_steps == 0 or
        dqn_summary.total_steps < dqn_summary.optimization_steps)
    {
        return error.DqnSmokeFailed;
    }

    const gcn_summary = try gcn_main.runSyntheticSmoke();
    if (gcn_summary.epochs_completed != 2 or !std.math.isFinite(gcn_summary.final_loss)) {
        return error.GcnSmokeFailed;
    }

    const char_lm_summary = try char_lm_main.trainCharLmSmoke();
    if (char_lm_summary.train_batches == 0 or
        !std.math.isFinite(char_lm_summary.initial_loss) or
        !std.math.isFinite(char_lm_summary.final_loss) or
        char_lm_summary.final_loss >= char_lm_summary.initial_loss)
    {
        return error.CharLmSmokeFailed;
    }

    const pendulum_summary = try pendulum_main.trainPendulumSmoke();
    if (pendulum_summary.train_batches == 0 or
        !std.math.isFinite(pendulum_summary.initial_loss) or
        !std.math.isFinite(pendulum_summary.final_loss) or
        !std.math.isFinite(pendulum_summary.rollout_rmse) or
        pendulum_summary.final_loss >= pendulum_summary.initial_loss or
        pendulum_summary.rollout_rmse >= 0.35)
    {
        return error.PendulumSmokeFailed;
    }

    const corridor_summary = try corridor_main.trainCorridorSmoke();
    if (corridor_summary.optimization_steps == 0 or
        !std.math.isFinite(corridor_summary.initial_eval_return) or
        !std.math.isFinite(corridor_summary.final_eval_return) or
        corridor_summary.final_eval_return <= corridor_summary.initial_eval_return or
        corridor_summary.final_success_rate < 0.9)
    {
        return error.CorridorSmokeFailed;
    }
}
