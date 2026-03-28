const std = @import("std");
const zg = @import("zigrad");

const Graph = zg.Graph;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn CharLmModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const nn = zg.nn(T);
        const depth = 2;
        const ParameterPack = struct {
            weights: [depth]*Tensor,
            biases: [depth]*Tensor,
        };

        weights: [depth]*Tensor = undefined,
        biases: [depth]*Tensor = undefined,
        context_len: usize,
        vocab_size: usize,
        hidden_size: usize,

        pub fn init(device: DeviceReference, context_len: usize, vocab_size: usize, hidden_size: usize) !Self {
            return initWithGraphAndSeed(device, context_len, vocab_size, hidden_size, null, null);
        }

        pub fn initWithGraph(
            device: DeviceReference,
            context_len: usize,
            vocab_size: usize,
            hidden_size: usize,
            graph: ?*Graph,
        ) !Self {
            return initWithGraphAndSeed(device, context_len, vocab_size, hidden_size, graph, null);
        }

        pub fn initWithGraphAndSeed(
            device: DeviceReference,
            context_len: usize,
            vocab_size: usize,
            hidden_size: usize,
            graph: ?*Graph,
            seed: ?u64,
        ) !Self {
            if (context_len == 0) return error.InvalidContextLength;
            if (vocab_size == 0) return error.InvalidVocabularySize;
            if (hidden_size == 0) return error.InvalidHiddenSize;

            const dims = [_]usize{ context_len * vocab_size, hidden_size, vocab_size };
            var self: Self = .{
                .context_len = context_len,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
            };

            var w_i: usize = 0;
            var b_i: usize = 0;
            errdefer {
                for (self.weights[0..w_i]) |weight| {
                    weight.release();
                    weight.deinit();
                }
                for (self.biases[0..b_i]) |bias| {
                    bias.release();
                    bias.deinit();
                }
            }

            inline for (&self.weights, &self.biases, 0..) |*weight, *bias, i| {
                const in_features = dims[i];
                const out_features = dims[i + 1];

                weight.* = if (seed) |value|
                    try makeDeterministicWeight(
                        std.heap.smp_allocator,
                        device,
                        graph,
                        out_features,
                        in_features,
                        value +% @as(u64, i + 1),
                        std.fmt.comptimePrint("weights.{d}", .{i}),
                    )
                else
                    try Tensor.random(
                        device,
                        &.{ out_features, in_features },
                        .{ .kaiming = in_features },
                        .{
                            .graph = graph,
                            .label = std.fmt.comptimePrint("weights.{d}", .{i}),
                            .requires_grad = true,
                            .acquired = true,
                        },
                    );
                w_i += 1;

                bias.* = try Tensor.zeros(
                    device,
                    &.{out_features},
                    .{
                        .graph = graph,
                        .label = std.fmt.comptimePrint("biases.{d}", .{i}),
                        .requires_grad = true,
                        .acquired = true,
                    },
                );
                b_i += 1;
            }

            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.weights, &self.biases) |weight, bias| {
                weight.release();
                weight.deinit();
                bias.release();
                bias.deinit();
            }
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            const batch_dim = x.data.shape.get(0);
            const other_dims = x.data.shape.crop(1, 0);
            const flattened_dim = zg.arrayutils.prod(other_dims);
            std.debug.assert(flattened_dim == self.context_len * self.vocab_size);

            const flat = try x.alias();
            flat.data._reshape(&.{ batch_dim, flattened_dim });
            flat.set_label("char_lm.flatten");
            errdefer flat.deinit();

            const hidden = try nn.linear(flat, self.weights[0], self.biases[0]);
            errdefer hidden.deinit();
            try nn.relu_(hidden);
            flat.soft_deinit();

            const logits = try nn.linear(hidden, self.weights[1], self.biases[1]);
            hidden.soft_deinit();
            return logits;
        }

        pub fn zero_grad(self: *Self) void {
            for (&self.weights) |*weight| weight.*.setup_grad(0) catch {};
            for (&self.biases) |*bias| bias.*.setup_grad(0) catch {};
        }

        pub fn attach_optimizer(self: *Self, optim: zg.Optimizer) !void {
            for (&self.weights, &self.biases) |*weight, *bias| {
                try optim.attach(weight.*);
                try optim.attach(bias.*);
            }
        }

        pub fn set_requires_grad(self: *Self, requires_grad: bool) void {
            for (&self.weights, &self.biases) |*weight, *bias| {
                if (requires_grad) {
                    weight.*.enable_grad();
                    bias.*.enable_grad();
                } else {
                    weight.*.disable_grad();
                    bias.*.disable_grad();
                }
            }
        }

        pub fn save(self: *Self, path: []const u8) !void {
            const allocator = std.heap.smp_allocator;
            var params = zg.LayerMap.init(allocator);
            defer params.deinit();

            for (&self.weights, &self.biases) |weight, bias| {
                try params.put(weight.get_label().?, weight, .{});
                try params.put(bias.get_label().?, bias, .{});
            }
            try params.save_to_file(path, allocator);
        }

        pub fn load(
            path: []const u8,
            device: DeviceReference,
            context_len: usize,
            vocab_size: usize,
            hidden_size: usize,
        ) !Self {
            const allocator = std.heap.smp_allocator;

            var params = try zg.LayerMap.load_from_file(path, allocator, device, .{
                .requires_grad = true,
                .acquired = true,
                .owning = false,
            });
            defer params.deinit();

            const pack = try params.extract(ParameterPack, "", .{});
            return .{
                .weights = pack.weights,
                .biases = pack.biases,
                .context_len = context_len,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
            };
        }

        fn makeDeterministicWeight(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: ?*Graph,
            rows: usize,
            cols: usize,
            seed: u64,
            label: []const u8,
        ) !*Tensor {
            const count = rows * cols;
            const host_values = try allocator.alloc(T, count);
            defer allocator.free(host_values);

            for (host_values, 0..) |*value, index| {
                const mixed = splitmix64(seed +% @as(u64, index));
                const normalized = (@as(f64, @floatFromInt(mixed % 10_000)) / 10_000.0) - 0.5;
                value.* = @as(T, @floatCast(normalized * 0.1));
            }

            return Tensor.from_slice(device, host_values, &.{ rows, cols }, .{
                .label = label,
                .requires_grad = true,
                .acquired = true,
                .graph = graph,
            });
        }

        fn splitmix64(state: u64) u64 {
            var z = state +% 0x9E3779B97F4A7C15;
            z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
            z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
            return z ^ (z >> 31);
        }
    };
}
