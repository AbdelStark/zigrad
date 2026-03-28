const std = @import("std");

const hello_world = @import("examples_hello_world_main");
const mnist_main = @import("examples_mnist_main");
const dqn_train = @import("examples_dqn_train");
const gcn_main = @import("examples_gcn_main");

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
}
