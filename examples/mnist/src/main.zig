/// Trains a neural network model on the MNIST dataset using a manual training loop.
const std = @import("std");
const zg = @import("zigrad");
const MnistDataset = @import("dataset.zig").MnistDataset;
const MnistModel = @import("model.zig").MnistModel;

const std_options = .{ .log_level = .info };
const T = f32;

pub const zigrad_settings: zg.Settings = .{
    .thread_safe = false,
    .logging = .{
        .level = .debug,
        .scopes = &.{
            .{ .scope = .zg_layer_map, .level = .info },
            .{ .scope = .zg_caching_allocator, .level = .info },
            .{ .scope = .zg_block_pool, .level = .info },
        },
    },
};

fn maybeWriteHostDiagnostics(cpu: *const zg.device.HostDevice, label: []const u8, include_telemetry: bool) void {
    _ = cpu.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = label,
        .include_telemetry = include_telemetry,
    }) catch {};
}

pub const RunConfig = struct {
    batch_size: usize = 64,
    num_epochs: usize = 3,
    verbose: ?bool = null,
    load_path: ?[]const u8 = "mnist.stz",
    save_path: ?[]const u8 = "mnist.stz",
};

pub const RunSummary = struct {
    epochs_completed: usize,
    train_batches: usize,
    train_accuracy: f32,
    test_accuracy: f32,
};

pub fn run_mnist(train_path: []const u8, test_path: []const u8) !void {
    _ = try run_mnist_with_config(train_path, test_path, .{});
}

pub fn run_mnist_with_config(train_path: []const u8, test_path: []const u8, config: RunConfig) !RunSummary {
    if (config.batch_size == 0) return error.InvalidBatchSize;

    const allocator = std.heap.smp_allocator;

    // use global graph for project
    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    maybeWriteHostDiagnostics(&cpu, "mnist:start", false);
    defer maybeWriteHostDiagnostics(&cpu, "mnist:summary", true);
    const device = cpu.reference();

    // std.debug.print("initializing device...", .{});
    // var gpu = zg.device.CudaDevice.init(0);
    // defer gpu.deinit();
    // std.debug.print("Done\n", .{});
    // const device = gpu.reference();

    var sgd = zg.optim.SGD.init(std.heap.smp_allocator, .{
        .lr = 0.01,
        .grad_clip_max_norm = 10.0,
        .grad_clip_delta = 1e-6,
        .grad_clip_enabled = false,
    });
    defer sgd.deinit();

    const optim = sgd.optimizer();

    std.debug.print("Loading model..\n", .{});

    var model = if (config.load_path) |load_path|
        MnistModel(T).load(load_path, device) catch |err| switch (err) {
            std.fs.File.OpenError.FileNotFound => try MnistModel(T).init(device),
            else => return err,
        }
    else
        try MnistModel(T).init(device);
    defer model.deinit();

    try model.attach_optimizer(optim);

    const verbose = config.verbose orelse ((std.posix.getenv("ZG_VERBOSE") orelse "0")[0] == '1');
    if (verbose) std.debug.print("Loading train data...\n", .{});
    const train_dataset = try MnistDataset(T).load(allocator, device, train_path, config.batch_size);

    var train_batches: usize = 0;

    // Train -------------------------------------------------------------------
    if (verbose) std.debug.print("Training...\n", .{});
    var timer = try std.time.Timer.start();
    var step_timer = try std.time.Timer.start();
    for (0..config.num_epochs) |epoch| {
        var total_loss: f64 = 0;
        for (train_dataset.images, train_dataset.labels, 0..) |image, label, i| {
            image.set_label("image_batch");
            label.set_label("label_batch");

            step_timer.reset();

            const output = try model.forward(image);
            defer output.deinit();

            const loss = try zg.loss.softmax_cross_entropy_loss(f32, output, label);
            defer loss.deinit();

            const loss_val = loss.get(0);

            try loss.backward();
            try optim.step();
            model.zero_grad();

            std.debug.assert(label.grad == null);
            std.debug.assert(image.grad == null);

            const t1 = @as(f64, @floatFromInt(step_timer.read()));
            const ms_per_sample = t1 / @as(f64, @floatFromInt(std.time.ns_per_ms * config.batch_size));
            total_loss += loss_val;
            train_batches += 1;

            if (verbose) std.debug.print("train_loss: {d:<5.5} [{d}/{d}] [ms/sample: {d}]\n", .{
                loss_val,
                i,
                train_dataset.images.len,
                ms_per_sample,
            });
        }
        const avg_loss = total_loss / @as(f32, @floatFromInt(train_dataset.images.len));
        if (verbose) std.debug.print("Epoch {d}: Avg Loss = {d:.4}\n", .{ epoch + 1, avg_loss });
    }
    const train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));

    //// Eval --------------------------------------------------------------------
    //// Eval on train set

    const train_eval = try eval_mnist(&model, train_dataset);
    const eval_train_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    train_dataset.deinit();

    // Eval on test set
    std.debug.print("Loading test data...\n", .{});
    const test_dataset = try MnistDataset(T).load(allocator, device, test_path, config.batch_size);
    defer test_dataset.deinit();
    timer.reset();
    const test_eval = try eval_mnist(&model, test_dataset);
    const eval_test_time_ms = @as(f64, @floatFromInt(timer.lap())) / @as(f64, @floatFromInt(std.time.ns_per_ms));
    std.debug.print("Test acc: {d:.2} (n={d})\n", .{ test_eval.acc * 100, test_eval.n });

    std.debug.print("Training complete ({d} epochs). [{d}ms]\n", .{ config.num_epochs, train_time_ms });
    std.debug.print("Eval train: {d:.2} (n={d}) [{d}ms]\n", .{ train_eval.acc * 100, train_eval.n, eval_train_time_ms });
    std.debug.print("Eval test: {d:.2} (n={d}) {d}ms\n", .{ test_eval.acc * 100, test_eval.n, eval_test_time_ms });

    if (config.save_path) |save_path| {
        std.debug.print("Saving model..\n", .{});
        try model.save(save_path);
    }

    return .{
        .epochs_completed = config.num_epochs,
        .train_batches = train_batches,
        .train_accuracy = train_eval.acc,
        .test_accuracy = test_eval.acc,
    };
}

pub fn main() !void {
    const smoke = (std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1';

    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    const data_sub_dir = std.posix.getenv("ZG_DATA_DIR") orelse "data";
    const train_name = if (smoke) "mnist_train_small.csv" else "mnist_train_full.csv";
    const test_name = if (smoke) "mnist_test_small.csv" else "mnist_test_full.csv";
    const train_path = try std.fmt.bufPrint(&buf1, "{s}/{s}", .{ data_sub_dir, train_name });
    const test_path = try std.fmt.bufPrint(&buf2, "{s}/{s}", .{ data_sub_dir, test_name });

    _ = try run_mnist_with_config(train_path, test_path, if (smoke)
        .{
            .batch_size = 16,
            .num_epochs = 1,
            .verbose = false,
            .load_path = null,
            .save_path = null,
        }
    else
        .{});
}

fn eval_mnist(model: *MnistModel(T), dataset: MnistDataset(T)) !struct { correct: f32, n: u32, acc: f32 } {
    zg.runtime.grad_enabled = false; // disable gradient tracking
    var n: u32 = 0;
    var correct: f32 = 0;
    for (dataset.images, dataset.labels) |image, label| {
        const output = try model.forward(image);
        defer output.deinit();
        const batch_n = output.data.shape.get(0);
        for (0..batch_n) |j| {
            const start = j * 10;
            const end = start + 10;
            const yh = std.mem.indexOfMax(T, output.data.data.raw[start..end]);
            const y = std.mem.indexOfMax(T, label.data.data.raw[start..end]);
            correct += if (yh == y) 1 else 0;
            n += 1;
        }
    }
    return .{ .correct = correct, .n = n, .acc = correct / @as(f32, @floatFromInt(n)) };
}

test run_mnist {
    const summary = run_mnist_with_config(
        "examples/mnist/data/mnist_train_small.csv",
        "examples/mnist/data/mnist_test_small.csv",
        .{
            .batch_size = 16,
            .num_epochs = 1,
            .verbose = false,
            .load_path = null,
            .save_path = null,
        },
    ) catch |err| switch (err) {
        std.fs.File.OpenError.FileNotFound => {
            std.log.warn("{s} error opening test file. Skipping `run_mnist` test.", .{@errorName(err)});
            return;
        },
        else => return err,
    };
    try std.testing.expectEqual(@as(usize, 1), summary.epochs_completed);
    try std.testing.expect(summary.train_batches > 0);
}
