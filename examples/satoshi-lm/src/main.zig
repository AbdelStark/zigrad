const std = @import("std");
const zg = @import("zigrad");

const SatoshiCorpus = @import("dataset.zig").SatoshiCorpus;
const SatoshiLmModel = @import("model.zig").SatoshiLmModel;

const std_options = .{ .log_level = .info };
const T = f32;

pub const RunConfig = struct {
    batch_size: usize = 64,
    num_epochs: usize = 5,
    context_len: usize = 64,
    hidden_size: usize = 128,
    learning_rate: T = 0.001,
    generate_chars: usize = 256,
    temperature: T = 0.8,
    stride: usize = 16,
    seed: u64 = 20090103,
    prompt: []const u8 = "The proof-of-work ",
    verbose: ?bool = null,
    load_path: ?[]const u8 = "satoshi-lm.stz",
    save_path: ?[]const u8 = "satoshi-lm.stz",
    device_request: ?zg.device.RuntimeDeviceRequest = null,
    val_split: T = 0.1,
};

pub const RunSummary = struct {
    epochs_completed: usize,
    train_batches: usize,
    initial_loss: T,
    final_loss: T,
    final_val_loss: T,
    vocab_size: usize,
    generated_bytes: usize,
};

pub fn runSatoshiLm() !void {
    _ = try runSatoshiLmWithConfig(.{});
}

pub fn trainSatoshiLmSmoke() !RunSummary {
    return runSatoshiLmWithConfig(.{
        .batch_size = 16,
        .num_epochs = 4,
        .context_len = 32,
        .hidden_size = 64,
        .learning_rate = 0.005,
        .generate_chars = 64,
        .temperature = 0.0,
        .stride = 256,
        .seed = 20090103,
        .prompt = "Bitcoin ",
        .load_path = null,
        .save_path = null,
        .verbose = false,
        .val_split = 0.1,
    });
}

pub fn runSatoshiLmWithConfig(config: RunConfig) !RunSummary {
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
        .label = "satoshi-lm:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "satoshi-lm:summary",
        .include_telemetry = true,
    }) catch {};
    const device = runtime_device.reference();

    const full_text = SatoshiCorpus.defaultCorpus();
    const split = SatoshiCorpus.splitText(full_text, config.val_split);

    var train_corpus = try SatoshiCorpus.init(allocator, split.train, config.context_len, config.batch_size, config.stride);
    defer train_corpus.deinit();

    var val_corpus = try SatoshiCorpus.initWithVocab(
        allocator,
        split.val,
        train_corpus.vocab,
        train_corpus.lookup,
        config.context_len,
        config.batch_size,
        config.stride,
    );
    defer val_corpus.deinit();

    var adam = zg.optim.Adam.init(allocator, .{
        .lr = config.learning_rate,
        .beta1 = 0.9,
        .beta2 = 0.999,
        .epsilon = 1e-8,
    });
    defer adam.deinit();
    const optimizer = adam.optimizer();

    var model = if (config.load_path) |load_path|
        SatoshiLmModel(T).load(
            load_path,
            device,
            config.context_len,
            train_corpus.vocabSize(),
            config.hidden_size,
        ) catch |err| switch (err) {
            std.fs.File.OpenError.FileNotFound => try SatoshiLmModel(T).initWithGraphAndSeed(
                device,
                config.context_len,
                train_corpus.vocabSize(),
                config.hidden_size,
                null,
                config.seed,
            ),
            else => return err,
        }
    else
        try SatoshiLmModel(T).initWithGraphAndSeed(
            device,
            config.context_len,
            train_corpus.vocabSize(),
            config.hidden_size,
            null,
            config.seed,
        );
    defer model.deinit();
    try model.attach_optimizer(optimizer);

    const verbose = config.verbose orelse ((std.posix.getenv("ZG_VERBOSE") orelse "0")[0] == '1');
    const initial_loss = try evaluateAverageLoss(&model, &train_corpus, device);

    std.debug.print("satoshi-lm: vocab={d}, train_samples={d}, val_samples={d}, batches/epoch={d}\n", .{
        train_corpus.vocabSize(),
        train_corpus.sampleCount(),
        val_corpus.sampleCount(),
        train_corpus.batchCount(),
    });

    var train_batches: usize = 0;
    var final_val_loss: T = 0;
    for (0..config.num_epochs) |epoch_index| {
        const epoch_result = try trainOneEpoch(&model, &train_corpus, device, optimizer);
        train_batches += epoch_result.batches;
        final_val_loss = try evaluateAverageLoss(&model, &val_corpus, device);
        if (verbose) {
            std.debug.print("epoch {d:>2}: train_loss={d:.4}  val_loss={d:.4}\n", .{
                epoch_index + 1,
                epoch_result.average_loss,
                final_val_loss,
            });
        }
    }

    const final_loss = try evaluateAverageLoss(&model, &train_corpus, device);
    const generated = if (config.generate_chars > 0)
        try generateWithTemperature(allocator, &model, &train_corpus, device, config.prompt, config.generate_chars, config.temperature, config.seed)
    else
        try allocator.dupe(u8, config.prompt);
    defer allocator.free(generated);

    std.debug.print(
        "\nsatoshi-lm: initial_loss={d:.4}, final_loss={d:.4}, val_loss={d:.4}, batches={d}\n",
        .{ initial_loss, final_loss, final_val_loss, train_batches },
    );
    std.debug.print("--- generated text ---\n{s}\n--- end ---\n", .{generated});

    if (config.save_path) |save_path| {
        try model.save(save_path);
    }

    return .{
        .epochs_completed = config.num_epochs,
        .train_batches = train_batches,
        .initial_loss = initial_loss,
        .final_loss = final_loss,
        .final_val_loss = final_val_loss,
        .vocab_size = train_corpus.vocabSize(),
        .generated_bytes = generated.len,
    };
}

pub fn main() !void {
    const smoke = (std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1';
    const prompt = std.posix.getenv("ZG_SATOSHI_LM_PROMPT") orelse "The proof-of-work ";

    _ = try runSatoshiLmWithConfig(if (smoke)
        .{
            .batch_size = 16,
            .num_epochs = 4,
            .context_len = 32,
            .hidden_size = 64,
            .learning_rate = 0.005,
            .generate_chars = 64,
            .temperature = 0.5,
            .stride = 256,
            .seed = 20090103,
            .prompt = prompt,
            .load_path = null,
            .save_path = null,
            .verbose = true,
        }
    else
        .{
            .prompt = prompt,
            .verbose = true,
        });
}

fn trainOneEpoch(
    model: *SatoshiLmModel(T),
    corpus: *const SatoshiCorpus,
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
    model: *SatoshiLmModel(T),
    corpus: *const SatoshiCorpus,
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

fn generateWithTemperature(
    allocator: std.mem.Allocator,
    model: *SatoshiLmModel(T),
    corpus: *const SatoshiCorpus,
    device: zg.DeviceReference,
    prompt: []const u8,
    generate_chars: usize,
    temperature: T,
    seed: u64,
) ![]u8 {
    const previous_grad_state = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_state;

    const total_len = prompt.len + generate_chars;
    const output = try allocator.alloc(u8, total_len);
    @memcpy(output[0..prompt.len], prompt);

    var rng_state = seed;

    for (prompt.len..total_len) |index| {
        const context = output[0..index];
        const input = try corpus.makePromptTensor(device, context);
        defer input.deinit();

        const logits = try model.forward(input);
        defer logits.deinit();

        const logits_host = try logits.to_host_owned(allocator);
        defer allocator.free(logits_host);

        const next_token = if (temperature <= 0.01)
            std.mem.indexOfMax(T, logits_host)
        else
            sampleFromLogits(logits_host, temperature, &rng_state);

        output[index] = corpus.byteForToken(next_token);
    }

    return output;
}

fn sampleFromLogits(logits: []const T, temperature: T, rng_state: *u64) usize {
    var max_val: T = logits[0];
    for (logits[1..]) |l| {
        if (l > max_val) max_val = l;
    }

    var probs: [256]T = undefined;
    var sum: T = 0;
    for (logits, 0..) |l, i| {
        probs[i] = @exp((l - max_val) / temperature);
        sum += probs[i];
    }

    rng_state.* = splitmix64(rng_state.*);
    var threshold = @as(T, @floatFromInt(rng_state.* % 1_000_000)) / 1_000_000.0;

    for (0..logits.len) |i| {
        threshold -= probs[i] / sum;
        if (threshold <= 0) return i;
    }
    return logits.len - 1;
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

test "train satoshi lm smoke" {
    const summary = try trainSatoshiLmSmoke();
    try std.testing.expectEqual(@as(usize, 4), summary.epochs_completed);
    try std.testing.expect(summary.train_batches > 0);
    try std.testing.expect(std.math.isFinite(summary.initial_loss));
    try std.testing.expect(std.math.isFinite(summary.final_loss));
    try std.testing.expect(std.math.isFinite(summary.final_val_loss));
    try std.testing.expect(summary.final_loss < summary.initial_loss);
}
