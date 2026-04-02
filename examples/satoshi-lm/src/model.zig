const std = @import("std");
const zg = @import("zigrad");

const Graph = zg.Graph;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn SatoshiLmModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const nn = zg.nn(T);

        pub const ParameterPack = struct {
            token_embedding: *Tensor,
            position_embedding: *Tensor,
            query_weights: *Tensor,
            query_bias: *Tensor,
            key_weights: *Tensor,
            key_bias: *Tensor,
            value_weights: *Tensor,
            value_bias: *Tensor,
            output_weights: *Tensor,
            output_bias: *Tensor,
        };

        token_embedding: *Tensor = undefined,
        position_embedding: *Tensor = undefined,
        query_weights: *Tensor = undefined,
        query_bias: *Tensor = undefined,
        key_weights: *Tensor = undefined,
        key_bias: *Tensor = undefined,
        value_weights: *Tensor = undefined,
        value_bias: *Tensor = undefined,
        output_weights: *Tensor = undefined,
        output_bias: *Tensor = undefined,
        causal_mask: *Tensor = undefined,
        position_selector: *Tensor = undefined,
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

            var self: Self = .{
                .context_len = context_len,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
            };

            self.token_embedding = try makeWeight(device, graph, hidden_size, vocab_size, seed, 1, "token_embedding");
            errdefer releaseOwnedTensor(self.token_embedding);
            self.position_embedding = try makeWeight(device, graph, context_len, hidden_size, seed, 2, "position_embedding");
            errdefer releaseOwnedTensor(self.position_embedding);

            self.query_weights = try makeWeight(device, graph, hidden_size, hidden_size, seed, 3, "attention.query_weights");
            errdefer releaseOwnedTensor(self.query_weights);
            self.query_bias = try makeBias(device, graph, hidden_size, "attention.query_bias");
            errdefer releaseOwnedTensor(self.query_bias);

            self.key_weights = try makeWeight(device, graph, hidden_size, hidden_size, seed, 4, "attention.key_weights");
            errdefer releaseOwnedTensor(self.key_weights);
            self.key_bias = try makeBias(device, graph, hidden_size, "attention.key_bias");
            errdefer releaseOwnedTensor(self.key_bias);

            self.value_weights = try makeWeight(device, graph, hidden_size, hidden_size, seed, 5, "attention.value_weights");
            errdefer releaseOwnedTensor(self.value_weights);
            self.value_bias = try makeBias(device, graph, hidden_size, "attention.value_bias");
            errdefer releaseOwnedTensor(self.value_bias);

            self.output_weights = try makeWeight(
                device,
                graph,
                vocab_size,
                hidden_size,
                seed,
                6,
                "output_weights",
            );
            errdefer releaseOwnedTensor(self.output_weights);
            self.output_bias = try makeBias(device, graph, vocab_size, "output_bias");
            errdefer releaseOwnedTensor(self.output_bias);

            self.causal_mask = try makeCausalMask(std.heap.smp_allocator, device, graph, context_len);
            errdefer releaseOwnedTensor(self.causal_mask);
            self.position_selector = try makePositionSelector(std.heap.smp_allocator, device, graph, context_len);
            errdefer releaseOwnedTensor(self.position_selector);

            return self;
        }

        pub fn deinit(self: *Self) void {
            releaseOwnedTensor(self.token_embedding);
            releaseOwnedTensor(self.position_embedding);
            releaseOwnedTensor(self.query_weights);
            releaseOwnedTensor(self.query_bias);
            releaseOwnedTensor(self.key_weights);
            releaseOwnedTensor(self.key_bias);
            releaseOwnedTensor(self.value_weights);
            releaseOwnedTensor(self.value_bias);
            releaseOwnedTensor(self.output_weights);
            releaseOwnedTensor(self.output_bias);
            releaseOwnedTensor(self.causal_mask);
            releaseOwnedTensor(self.position_selector);
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *Tensor) !*Tensor {
            std.debug.assert(x.data.shape.len == 3);
            std.debug.assert(x.data.shape.get(1) == self.context_len);
            std.debug.assert(x.data.shape.get(2) == self.vocab_size);

            const scale = @as(T, @floatCast(1.0 / std.math.sqrt(@as(f64, @floatFromInt(self.hidden_size)))));
            const batch_dim = x.data.shape.get(0);

            const token_embeddings = try nn.linear(x, self.token_embedding, null);
            errdefer token_embeddings.deinit();

            const embeddings = try token_embeddings.add(self.position_embedding);
            token_embeddings.soft_deinit();
            errdefer embeddings.deinit();

            const queries = try nn.linear(embeddings, self.query_weights, self.query_bias);
            errdefer queries.deinit();
            const keys = try nn.linear(embeddings, self.key_weights, self.key_bias);
            errdefer keys.deinit();
            const values = try nn.linear(embeddings, self.value_weights, self.value_bias);
            errdefer values.deinit();

            const scores = try queries.bmm(keys, .{
                .trans_b = true,
                .alpha = scale,
            });
            queries.soft_deinit();
            keys.soft_deinit();
            errdefer scores.deinit();

            const masked_scores = try scores.add(self.causal_mask);
            scores.soft_deinit();
            errdefer masked_scores.deinit();

            const attention = try zg.loss.softmax(T, masked_scores, masked_scores.data.shape.len - 1, x.device);
            masked_scores.soft_deinit();
            errdefer attention.deinit();

            const attended = try attention.bmm(values, .{});
            attention.soft_deinit();
            values.soft_deinit();
            errdefer attended.deinit();

            const mixed = try attended.add(embeddings);
            attended.soft_deinit();
            embeddings.soft_deinit();
            errdefer mixed.deinit();

            const summary = try self.position_selector.bmm(mixed, .{});
            mixed.soft_deinit();
            errdefer summary.deinit();

            const flat = try summary.alias();
            flat.data._reshape(&.{ batch_dim, self.hidden_size });
            flat.set_label("satoshi_lm.summary");
            errdefer flat.deinit();

            const logits = try nn.linear(flat, self.output_weights, self.output_bias);
            flat.soft_deinit();
            summary.soft_deinit();
            return logits;
        }

        pub fn zero_grad(self: *Self) void {
            self.token_embedding.setup_grad(0) catch {};
            self.position_embedding.setup_grad(0) catch {};
            self.query_weights.setup_grad(0) catch {};
            self.query_bias.setup_grad(0) catch {};
            self.key_weights.setup_grad(0) catch {};
            self.key_bias.setup_grad(0) catch {};
            self.value_weights.setup_grad(0) catch {};
            self.value_bias.setup_grad(0) catch {};
            self.output_weights.setup_grad(0) catch {};
            self.output_bias.setup_grad(0) catch {};
        }

        pub fn attach_optimizer(self: *Self, optim: zg.Optimizer) !void {
            try optim.attach(self.token_embedding);
            try optim.attach(self.position_embedding);
            try optim.attach(self.query_weights);
            try optim.attach(self.query_bias);
            try optim.attach(self.key_weights);
            try optim.attach(self.key_bias);
            try optim.attach(self.value_weights);
            try optim.attach(self.value_bias);
            try optim.attach(self.output_weights);
            try optim.attach(self.output_bias);
        }

        pub fn set_requires_grad(self: *Self, requires_grad: bool) void {
            setTensorRequiresGrad(self.token_embedding, requires_grad);
            setTensorRequiresGrad(self.position_embedding, requires_grad);
            setTensorRequiresGrad(self.query_weights, requires_grad);
            setTensorRequiresGrad(self.query_bias, requires_grad);
            setTensorRequiresGrad(self.key_weights, requires_grad);
            setTensorRequiresGrad(self.key_bias, requires_grad);
            setTensorRequiresGrad(self.value_weights, requires_grad);
            setTensorRequiresGrad(self.value_bias, requires_grad);
            setTensorRequiresGrad(self.output_weights, requires_grad);
            setTensorRequiresGrad(self.output_bias, requires_grad);
        }

        pub fn save(self: *Self, path: []const u8) !void {
            const allocator = std.heap.smp_allocator;
            var params = zg.LayerMap.init(allocator);
            defer params.deinit();

            try writeParameters(&params, self.parameterPack());
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
            return fromParameterPack(device, context_len, vocab_size, hidden_size, pack, null);
        }

        pub fn fromParameterPack(
            device: DeviceReference,
            context_len: usize,
            vocab_size: usize,
            hidden_size: usize,
            pack: ParameterPack,
            graph: ?*Graph,
        ) !Self {
            const causal_mask = try makeCausalMask(std.heap.smp_allocator, device, graph, context_len);
            errdefer releaseOwnedTensor(causal_mask);
            const position_selector = try makePositionSelector(std.heap.smp_allocator, device, graph, context_len);
            errdefer releaseOwnedTensor(position_selector);

            return .{
                .token_embedding = pack.token_embedding,
                .position_embedding = pack.position_embedding,
                .query_weights = pack.query_weights,
                .query_bias = pack.query_bias,
                .key_weights = pack.key_weights,
                .key_bias = pack.key_bias,
                .value_weights = pack.value_weights,
                .value_bias = pack.value_bias,
                .output_weights = pack.output_weights,
                .output_bias = pack.output_bias,
                .causal_mask = causal_mask,
                .position_selector = position_selector,
                .context_len = context_len,
                .vocab_size = vocab_size,
                .hidden_size = hidden_size,
            };
        }

        fn parameterPack(self: *Self) ParameterPack {
            return .{
                .token_embedding = self.token_embedding,
                .position_embedding = self.position_embedding,
                .query_weights = self.query_weights,
                .query_bias = self.query_bias,
                .key_weights = self.key_weights,
                .key_bias = self.key_bias,
                .value_weights = self.value_weights,
                .value_bias = self.value_bias,
                .output_weights = self.output_weights,
                .output_bias = self.output_bias,
            };
        }

        fn writeParameters(params: *zg.LayerMap, pack: ParameterPack) !void {
            try params.put("token_embedding", pack.token_embedding, .{});
            try params.put("position_embedding", pack.position_embedding, .{});
            try params.put("query_weights", pack.query_weights, .{});
            try params.put("query_bias", pack.query_bias, .{});
            try params.put("key_weights", pack.key_weights, .{});
            try params.put("key_bias", pack.key_bias, .{});
            try params.put("value_weights", pack.value_weights, .{});
            try params.put("value_bias", pack.value_bias, .{});
            try params.put("output_weights", pack.output_weights, .{});
            try params.put("output_bias", pack.output_bias, .{});
        }

        fn releaseOwnedTensor(tensor: *Tensor) void {
            tensor.release();
            tensor.deinit();
        }

        fn setTensorRequiresGrad(tensor: *Tensor, requires_grad: bool) void {
            if (requires_grad) {
                tensor.enable_grad();
            } else {
                tensor.disable_grad();
            }
        }

        fn makeWeight(
            device: DeviceReference,
            graph: ?*Graph,
            rows: usize,
            cols: usize,
            seed: ?u64,
            seed_offset: u64,
            label: []const u8,
        ) !*Tensor {
            return if (seed) |value|
                makeDeterministicWeight(
                    std.heap.smp_allocator,
                    device,
                    graph,
                    rows,
                    cols,
                    value +% seed_offset,
                    label,
                )
            else
                Tensor.random(
                    device,
                    &.{ rows, cols },
                    .{ .kaiming = cols },
                    .{
                        .graph = graph,
                        .label = label,
                        .requires_grad = true,
                        .acquired = true,
                    },
                );
        }

        fn makeBias(
            device: DeviceReference,
            graph: ?*Graph,
            len: usize,
            label: []const u8,
        ) !*Tensor {
            return Tensor.zeros(
                device,
                &.{len},
                .{
                    .graph = graph,
                    .label = label,
                    .requires_grad = true,
                    .acquired = true,
                },
            );
        }

        fn makeCausalMask(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: ?*Graph,
            context_len: usize,
        ) !*Tensor {
            const count = context_len * context_len;
            const host_values = try allocator.alloc(T, count);
            defer allocator.free(host_values);

            for (host_values, 0..) |*value, index| {
                const row = index / context_len;
                const column = index % context_len;
                value.* = if (column > row) @as(T, -1.0e4) else @as(T, 0);
            }

            return Tensor.from_slice(device, host_values, &.{ 1, context_len, context_len }, .{
                .label = "attention.causal_mask",
                .requires_grad = false,
                .acquired = true,
                .graph = graph,
            });
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

        fn makePositionSelector(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: ?*Graph,
            context_len: usize,
        ) !*Tensor {
            const host_values = try allocator.alloc(T, context_len);
            defer allocator.free(host_values);
            @memset(host_values, 0);
            host_values[context_len - 1] = 1;

            return Tensor.from_slice(device, host_values, &.{ 1, context_len }, .{
                .label = "attention.last_token_selector",
                .requires_grad = false,
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
