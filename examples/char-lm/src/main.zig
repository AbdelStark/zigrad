const std = @import("std");
const zg = @import("zigrad");

const CharCorpus = @import("dataset.zig").CharCorpus;
const CharLmModel = @import("model.zig").CharLmModel;

const std_options = .{ .log_level = .info };
const T = f32;

pub const RunConfig = struct {
    batch_size: usize = 16,
    num_epochs: usize = 24,
    context_len: usize = 16,
    hidden_size: usize = 64,
    learning_rate: T = 0.01,
    generate_chars: usize = 64,
    seed: u64 = 20260328,
    prompt: []const u8 = "zigrad ",
    verbose: ?bool = null,
    load_path: ?[]const u8 = "char-lm.stz",
    save_path: ?[]const u8 = "char-lm.stz",
    device_request: ?zg.device.RuntimeDeviceRequest = null,
};

pub const RunSummary = struct {
    epochs_completed: usize,
    train_batches: usize,
    initial_loss: T,
    final_loss: T,
    vocab_size: usize,
    generated_bytes: usize,
};

pub fn runCharLm() !void {
    _ = try runCharLmWithConfig(.{});
}

pub fn trainCharLmSmoke() !RunSummary {
    return runCharLmWithConfig(.{
        .batch_size = 8,
        .num_epochs = 12,
        .context_len = 12,
        .hidden_size = 48,
        .learning_rate = 0.02,
        .generate_chars = 24,
        .seed = 7,
        .prompt = "zigrad ",
        .load_path = null,
        .save_path = null,
        .verbose = false,
    });
}

pub fn runCharLmWithConfig(config: RunConfig) !RunSummary {
    if (config.batch_size == 0) return error.InvalidBatchSize;
    if (config.num_epochs == 0) return error.InvalidEpochCount;
    if (config.context_len == 0) return error.InvalidContextLength;
    if (config.hidden_size == 0) return error.InvalidHiddenSize;

    const allocator = std.heap.smp_allocator;

    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    var runtime_device = try zg.device.initRuntimeDevice(config.device_request, .{ .allow_cuda = true });
    defer runtime_device.deinit();
    _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "char-lm:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "char-lm:summary",
        .include_telemetry = true,
    }) catch {};
    const device = runtime_device.reference();

    var corpus = try CharCorpus.init(allocator, CharCorpus.defaultCorpus(), config.context_len, config.batch_size);
    defer corpus.deinit();

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = config.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();

    var model = if (config.load_path) |load_path|
        CharLmModel(T).load(
            load_path,
            device,
            config.context_len,
            corpus.vocabSize(),
            config.hidden_size,
        ) catch |err| switch (err) {
            std.fs.File.OpenError.FileNotFound => try CharLmModel(T).initWithGraphAndSeed(
                device,
                config.context_len,
                corpus.vocabSize(),
                config.hidden_size,
                null,
                config.seed,
            ),
            else => return err,
        }
    else
        try CharLmModel(T).initWithGraphAndSeed(
            device,
            config.context_len,
            corpus.vocabSize(),
            config.hidden_size,
            null,
            config.seed,
        );
    defer model.deinit();
    try model.attach_optimizer(optimizer);

    const verbose = config.verbose orelse ((std.posix.getenv("ZG_VERBOSE") orelse "0")[0] == '1');
    const initial_loss = try evaluateAverageLoss(&model, &corpus, device);

    var train_batches: usize = 0;
    for (0..config.num_epochs) |epoch_index| {
        const epoch_result = try trainOneEpoch(&model, &corpus, device, optimizer);
        train_batches += epoch_result.batches;
        if (verbose) {
            std.debug.print("epoch {d:>2}: avg_loss={d:.4}\n", .{
                epoch_index + 1,
                epoch_result.average_loss,
            });
        }
    }

    const final_loss = try evaluateAverageLoss(&model, &corpus, device);
    const generated = if (config.generate_chars > 0)
        try generateGreedy(allocator, &model, &corpus, device, config.prompt, config.generate_chars)
    else
        try allocator.dupe(u8, config.prompt);
    defer allocator.free(generated);

    std.debug.print(
        "char-lm: vocab={d}, initial_loss={d:.4}, final_loss={d:.4}, batches={d}\n",
        .{ corpus.vocabSize(), initial_loss, final_loss, train_batches },
    );
    std.debug.print("generated:\n{s}\n", .{generated});

    if (config.save_path) |save_path| {
        try model.save(save_path);
    }

    return .{
        .epochs_completed = config.num_epochs,
        .train_batches = train_batches,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .vocab_size = corpus.vocabSize(),
        .generated_bytes = generated.len,
    };
}

pub fn main() !void {
    const smoke = (std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1';
    const prompt = std.posix.getenv("ZG_CHAR_LM_PROMPT") orelse "zigrad ";

    _ = try runCharLmWithConfig(if (smoke)
        .{
            .batch_size = 8,
            .num_epochs = 12,
            .context_len = 12,
            .hidden_size = 48,
            .learning_rate = 0.02,
            .generate_chars = 24,
            .seed = 7,
            .prompt = prompt,
            .load_path = null,
            .save_path = null,
            .verbose = false,
        }
    else
        .{
            .prompt = prompt,
        });
}

fn trainOneEpoch(
    model: *CharLmModel(T),
    corpus: *const CharCorpus,
    device: zg.DeviceReference,
    optimizer: zg.Optimizer,
) !struct { average_loss: T, batches: usize } {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = true;
    defer zg.runtime.grad_enabled = previous_grad_state;

    var loss_sum: T = 0;
    const batch_count = corpus.batchCount();

    for (0..batch_count) |batch_index| {
        var batch = try corpus.makeBatch(device, batch_index);
        defer batch.deinit();

        const logits = try model.forward(batch.inputs);
        defer logits.deinit();

        const loss = try zg.loss.softmax_cross_entropy_loss(T, logits, batch.labels);
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
    model: *CharLmModel(T),
    corpus: *const CharCorpus,
    device: zg.DeviceReference,
) !T {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    var loss_sum: T = 0;
    const batch_count = corpus.batchCount();

    for (0..batch_count) |batch_index| {
        var batch = try corpus.makeBatch(device, batch_index);
        defer batch.deinit();

        const logits = try model.forward(batch.inputs);
        defer logits.deinit();

        const loss = try zg.loss.softmax_cross_entropy_loss(T, logits, batch.labels);
        defer loss.deinit();

        loss_sum += loss.get(0);
    }

    return loss_sum / @as(T, @floatFromInt(batch_count));
}

fn generateGreedy(
    allocator: std.mem.Allocator,
    model: *CharLmModel(T),
    corpus: *const CharCorpus,
    device: zg.DeviceReference,
    prompt: []const u8,
    generate_chars: usize,
) ![]u8 {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const total_len = prompt.len + generate_chars;
    const output = try allocator.alloc(u8, total_len);
    @memcpy(output[0..prompt.len], prompt);

    for (prompt.len..total_len) |index| {
        const context = output[0..index];
        const input = try corpus.makePromptTensor(device, context);
        defer input.deinit();

        const logits = try model.forward(input);
        defer logits.deinit();

        const logits_host = try logits.to_host_owned(allocator);
        defer allocator.free(logits_host);

        const next_token = std.mem.indexOfMax(T, logits_host);
        output[index] = corpus.byteForToken(next_token);
    }

    return output;
}

test "train char lm smoke" {
    const summary = try trainCharLmSmoke();
    try std.testing.expectEqual(@as(usize, 12), summary.epochs_completed);
    try std.testing.expect(summary.train_batches > 0);
    try std.testing.expect(std.math.isFinite(summary.initial_loss));
    try std.testing.expect(std.math.isFinite(summary.final_loss));
    try std.testing.expect(summary.final_loss < summary.initial_loss);
}
