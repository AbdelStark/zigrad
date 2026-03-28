const std = @import("std");
const zg = @import("zigrad");

const DeviceReference = zg.DeviceReference;
const Graph = zg.Graph;
const NDTensor = zg.NDTensor;

pub fn CorridorControlModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const nn = zg.nn(T);
        const depth = 3;

        const ParameterPack = struct {
            weights: [depth]*Tensor,
            biases: [depth]*Tensor,
        };

        weights: [depth]*Tensor = undefined,
        biases: [depth]*Tensor = undefined,
        input_size: usize,
        hidden_size: usize,
        output_size: usize,

        pub fn init(
            device: DeviceReference,
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
        ) !Self {
            return initWithGraphAndSeed(device, input_size, hidden_size, output_size, null, null);
        }

        pub fn initWithGraph(
            device: DeviceReference,
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            graph: ?*Graph,
        ) !Self {
            return initWithGraphAndSeed(device, input_size, hidden_size, output_size, graph, null);
        }

        pub fn initWithGraphAndSeed(
            device: DeviceReference,
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
            graph: ?*Graph,
            seed: ?u64,
        ) !Self {
            if (input_size == 0) return error.InvalidInputSize;
            if (hidden_size == 0) return error.InvalidHiddenSize;
            if (output_size == 0) return error.InvalidOutputSize;

            const dims = [_]usize{ input_size, hidden_size, hidden_size, output_size };
            var self: Self = .{
                .input_size = input_size,
                .hidden_size = hidden_size,
                .output_size = output_size,
            };

            var weight_count: usize = 0;
            var bias_count: usize = 0;
            errdefer {
                for (self.weights[0..weight_count]) |weight| {
                    weight.release();
                    weight.deinit();
                }
                for (self.biases[0..bias_count]) |bias| {
                    bias.release();
                    bias.deinit();
                }
            }

            inline for (&self.weights, &self.biases, 0..) |*weight, *bias, index| {
                const in_features = dims[index];
                const out_features = dims[index + 1];

                weight.* = if (seed) |value|
                    try makeDeterministicWeight(
                        std.heap.smp_allocator,
                        device,
                        graph,
                        out_features,
                        in_features,
                        value +% @as(u64, index + 1),
                        std.fmt.comptimePrint("corridor.weights.{d}", .{index}),
                    )
                else
                    try Tensor.random(
                        device,
                        &.{ out_features, in_features },
                        .{ .kaiming = in_features },
                        .{
                            .graph = graph,
                            .label = std.fmt.comptimePrint("corridor.weights.{d}", .{index}),
                            .requires_grad = true,
                            .acquired = true,
                        },
                    );
                weight_count += 1;

                bias.* = try Tensor.zeros(
                    device,
                    &.{out_features},
                    .{
                        .graph = graph,
                        .label = std.fmt.comptimePrint("corridor.biases.{d}", .{index}),
                        .requires_grad = true,
                        .acquired = true,
                    },
                );
                bias_count += 1;
            }

            return self;
        }

        pub fn clone(self: *const Self) !Self {
            var result: Self = .{
                .input_size = self.input_size,
                .hidden_size = self.hidden_size,
                .output_size = self.output_size,
            };

            var weight_count: usize = 0;
            var bias_count: usize = 0;
            errdefer {
                for (result.weights[0..weight_count]) |weight| {
                    weight.release();
                    weight.deinit();
                }
                for (result.biases[0..bias_count]) |bias| {
                    bias.release();
                    bias.deinit();
                }
            }

            inline for (0..depth) |index| {
                result.weights[index] = try self.weights[index].clone();
                result.weights[index].detach();
                result.weights[index].acquire();
                weight_count += 1;

                result.biases[index] = try self.biases[index].clone();
                result.biases[index].detach();
                result.biases[index].acquire();
                bias_count += 1;
            }

            return result;
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
            std.debug.assert(flattened_dim == self.input_size);

            const flat = try x.alias();
            flat.data._reshape(&.{ batch_dim, flattened_dim });
            flat.set_label("corridor.flatten");
            errdefer flat.deinit();

            const hidden0 = try nn.linear(flat, self.weights[0], self.biases[0]);
            errdefer hidden0.deinit();
            try nn.relu_(hidden0);
            flat.soft_deinit();

            const hidden1 = try nn.linear(hidden0, self.weights[1], self.biases[1]);
            errdefer hidden1.deinit();
            try nn.relu_(hidden1);
            hidden0.soft_deinit();

            const output = try nn.linear(hidden1, self.weights[2], self.biases[2]);
            hidden1.soft_deinit();
            return output;
        }

        pub fn zero_grad(self: *Self) void {
            for (&self.weights) |*weight| weight.*.setup_grad(0) catch {};
            for (&self.biases) |*bias| bias.*.setup_grad(0) catch {};
        }

        pub fn attach_optimizer(self: *Self, optimizer: zg.Optimizer) !void {
            for (&self.weights, &self.biases) |*weight, *bias| {
                try optimizer.attach(weight.*);
                try optimizer.attach(bias.*);
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

        pub fn soft_update_from(self: *Self, other: *const Self, tau: T) !void {
            std.debug.assert(self.input_size == other.input_size);
            std.debug.assert(self.hidden_size == other.hidden_size);
            std.debug.assert(self.output_size == other.output_size);
            std.debug.assert(tau >= 0.0 and tau <= 1.0);

            for (&self.weights, &self.biases, &other.weights, &other.biases) |weight_dst, bias_dst, weight_src, bias_src| {
                const device = weight_src.device;
                std.debug.assert(device.is_compatible(weight_dst.device));

                weight_dst.data._scale(1.0 - tau, device);
                var weight_delta = try weight_src.data.copy(device);
                weight_delta._scale(tau, device);
                try weight_dst.data._add(weight_delta, device);
                weight_delta.deinit(device);

                bias_dst.data._scale(1.0 - tau, device);
                var bias_delta = try bias_src.data.copy(device);
                bias_delta._scale(tau, device);
                try bias_dst.data._add(bias_delta, device);
                bias_delta.deinit(device);
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
            input_size: usize,
            hidden_size: usize,
            output_size: usize,
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
                .input_size = input_size,
                .hidden_size = hidden_size,
                .output_size = output_size,
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
                value.* = @as(T, @floatCast(normalized * 0.12));
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
