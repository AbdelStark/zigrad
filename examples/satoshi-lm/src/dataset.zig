const std = @import("std");
const zg = @import("zigrad");

const Tensor = zg.NDTensor(f32);

pub const SatoshiCorpus = struct {
    allocator: std.mem.Allocator,
    vocab: []u8,
    lookup: [256]usize,
    token_ids: []usize,
    context_len: usize,
    batch_size: usize,
    stride: usize,

    pub const Batch = struct {
        inputs: *Tensor,
        labels: *Tensor,
        size: usize,

        pub fn deinit(self: *Batch) void {
            self.inputs.deinit();
            self.labels.deinit();
            self.* = undefined;
        }
    };

    pub fn defaultCorpus() []const u8 {
        return @embedFile("satoshi-all-text.txt");
    }

    /// Split text at the nearest newline to `(1 - val_ratio)` of the total length.
    pub fn splitText(text: []const u8, val_ratio: f32) struct { train: []const u8, val: []const u8 } {
        const split_pos = @as(usize, @intFromFloat(@as(f32, @floatFromInt(text.len)) * (1.0 - val_ratio)));
        var boundary = split_pos;
        while (boundary < text.len and text[boundary] != '\n') {
            boundary += 1;
        }
        if (boundary < text.len) boundary += 1;
        return .{
            .train = text[0..boundary],
            .val = if (boundary < text.len) text[boundary..] else text[split_pos..],
        };
    }

    pub fn init(
        allocator: std.mem.Allocator,
        text: []const u8,
        context_len: usize,
        batch_size: usize,
        stride: usize,
    ) !SatoshiCorpus {
        if (context_len == 0) return error.InvalidContextLength;
        if (batch_size == 0) return error.InvalidBatchSize;
        if (stride == 0) return error.InvalidStride;
        if (text.len <= context_len) return error.CorpusTooSmall;

        const sentinel = std.math.maxInt(usize);
        var lookup = [_]usize{sentinel} ** 256;
        var vocab_buf: [256]u8 = undefined;
        var vocab_len: usize = 0;

        for (text) |byte| {
            if (lookup[byte] != sentinel) continue;
            lookup[byte] = vocab_len;
            vocab_buf[vocab_len] = byte;
            vocab_len += 1;
        }

        const vocab = try allocator.dupe(u8, vocab_buf[0..vocab_len]);
        errdefer allocator.free(vocab);

        const token_ids = try allocator.alloc(usize, text.len);
        errdefer allocator.free(token_ids);
        for (text, 0..) |byte, index| {
            token_ids[index] = lookup[byte];
        }

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .lookup = lookup,
            .token_ids = token_ids,
            .context_len = context_len,
            .batch_size = batch_size,
            .stride = stride,
        };
    }

    /// Create a corpus that shares vocabulary with another (for validation splits).
    pub fn initWithVocab(
        allocator: std.mem.Allocator,
        text: []const u8,
        source_vocab: []const u8,
        source_lookup: [256]usize,
        context_len: usize,
        batch_size: usize,
        stride: usize,
    ) !SatoshiCorpus {
        if (context_len == 0) return error.InvalidContextLength;
        if (batch_size == 0) return error.InvalidBatchSize;
        if (stride == 0) return error.InvalidStride;
        if (text.len <= context_len) return error.CorpusTooSmall;

        const vocab = try allocator.dupe(u8, source_vocab);
        errdefer allocator.free(vocab);

        const sentinel = std.math.maxInt(usize);
        const space_token = if (source_lookup[' '] != sentinel) source_lookup[' '] else 0;

        const token_ids = try allocator.alloc(usize, text.len);
        errdefer allocator.free(token_ids);
        for (text, 0..) |byte, index| {
            token_ids[index] = if (source_lookup[byte] != sentinel) source_lookup[byte] else space_token;
        }

        return .{
            .allocator = allocator,
            .vocab = vocab,
            .lookup = source_lookup,
            .token_ids = token_ids,
            .context_len = context_len,
            .batch_size = batch_size,
            .stride = stride,
        };
    }

    pub fn deinit(self: *SatoshiCorpus) void {
        self.allocator.free(self.vocab);
        self.allocator.free(self.token_ids);
        self.* = undefined;
    }

    pub fn sampleCount(self: *const SatoshiCorpus) usize {
        const available = self.token_ids.len - self.context_len;
        return (available + self.stride - 1) / self.stride;
    }

    pub fn batchCount(self: *const SatoshiCorpus) usize {
        const samples = self.sampleCount();
        return (samples + self.batch_size - 1) / self.batch_size;
    }

    pub fn vocabSize(self: *const SatoshiCorpus) usize {
        return self.vocab.len;
    }

    pub fn tokenForByte(self: *const SatoshiCorpus, byte: u8) usize {
        const sentinel = std.math.maxInt(usize);
        if (self.lookup[byte] != sentinel) return self.lookup[byte];
        if (self.lookup[' '] != sentinel) return self.lookup[' '];
        return 0;
    }

    pub fn byteForToken(self: *const SatoshiCorpus, token: usize) u8 {
        return self.vocab[token];
    }

    pub fn makeBatch(self: *const SatoshiCorpus, device: zg.DeviceReference, batch_index: usize) !Batch {
        const sample_count = self.sampleCount();
        const start = batch_index * self.batch_size;
        if (start >= sample_count) return error.InvalidBatchIndex;

        const actual_batch_size = @min(self.batch_size, sample_count - start);
        const vocab_size = self.vocab.len;

        const input_values = try self.allocator.alloc(f32, actual_batch_size * self.context_len * vocab_size);
        defer self.allocator.free(input_values);
        @memset(input_values, 0);

        const label_values = try self.allocator.alloc(f32, actual_batch_size * vocab_size);
        defer self.allocator.free(label_values);
        @memset(label_values, 0);

        for (0..actual_batch_size) |row| {
            const sample_start = (start + row) * self.stride;
            for (0..self.context_len) |column| {
                const token = self.token_ids[sample_start + column];
                input_values[((row * self.context_len + column) * vocab_size) + token] = 1;
            }

            const label_token = self.token_ids[sample_start + self.context_len];
            label_values[(row * vocab_size) + label_token] = 1;
        }

        const inputs = try Tensor.from_slice(device, input_values, &.{ actual_batch_size, self.context_len, vocab_size }, .{});
        errdefer inputs.deinit();
        const labels = try Tensor.from_slice(device, label_values, &.{ actual_batch_size, vocab_size }, .{});

        return .{
            .inputs = inputs,
            .labels = labels,
            .size = actual_batch_size,
        };
    }

    pub fn makePromptTensor(
        self: *const SatoshiCorpus,
        device: zg.DeviceReference,
        prompt: []const u8,
    ) !*Tensor {
        const vocab_size = self.vocab.len;
        const input_values = try self.allocator.alloc(f32, self.context_len * vocab_size);
        defer self.allocator.free(input_values);
        @memset(input_values, 0);

        const window = if (prompt.len > self.context_len)
            prompt[prompt.len - self.context_len ..]
        else
            prompt;

        const pad_len = self.context_len - window.len;
        const pad_token = self.tokenForByte(' ');

        for (0..pad_len) |column| {
            input_values[(column * vocab_size) + pad_token] = 1;
        }
        for (window, 0..) |byte, index| {
            const token = self.tokenForByte(byte);
            input_values[((pad_len + index) * vocab_size) + token] = 1;
        }

        return Tensor.from_slice(device, input_values, &.{ 1, self.context_len, vocab_size }, .{});
    }
};
