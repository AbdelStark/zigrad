const std = @import("std");
const zg = @import("zigrad");
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn MnistBenchmarkModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);
        const nn = zg.nn(T);

        const dims = [_]usize{ 784, 128, 64, 10 };
        const depth = dims.len - 1;

        weights: [depth]*Tensor = undefined,
        biases: [depth]*Tensor = undefined,

        pub fn init(allocator: std.mem.Allocator, device: DeviceReference, graph: *zg.Graph, seed: u64) !Self {
            var self: Self = .{};
            var w_count: usize = 0;
            var b_count: usize = 0;
            errdefer {
                for (self.weights[0..w_count]) |weight| {
                    weight.release();
                    weight.deinit();
                }
                for (self.biases[0..b_count]) |bias| {
                    bias.release();
                    bias.deinit();
                }
            }

            inline for (0..depth) |i| {
                const in_features = dims[i];
                const out_features = dims[i + 1];

                self.weights[i] = try makeWeight(
                    allocator,
                    device,
                    graph,
                    out_features,
                    in_features,
                    seed +% @as(u64, i + 1),
                    std.fmt.comptimePrint("bench.weights.{d}", .{i}),
                );
                w_count += 1;

                self.biases[i] = try Tensor.zeros(device, &.{out_features}, .{
                    .label = std.fmt.comptimePrint("bench.biases.{d}", .{i}),
                    .requires_grad = true,
                    .acquired = true,
                    .graph = graph,
                });
                b_count += 1;
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

            const flat = try x.alias();
            flat.data._reshape(&.{ batch_dim, flattened_dim });
            flat.set_label("benchmark.flatten");
            errdefer flat.deinit();

            const z0 = try nn.linear(flat, self.weights[0], self.biases[0]);
            errdefer z0.deinit();
            try nn.relu_(z0);
            flat.soft_deinit();

            const z1 = try nn.linear(z0, self.weights[1], self.biases[1]);
            errdefer z1.deinit();
            try nn.relu_(z1);
            z0.soft_deinit();

            const z2 = try nn.linear(z1, self.weights[2], self.biases[2]);
            z1.soft_deinit();
            return z2;
        }

        pub fn zeroGrad(self: *Self) void {
            for (&self.weights) |*weight| weight.*.setup_grad(0) catch {};
            for (&self.biases) |*bias| bias.*.setup_grad(0) catch {};
        }

        pub fn attachOptimizer(self: *Self, optimizer: zg.Optimizer) !void {
            for (&self.weights, &self.biases) |*weight, *bias| {
                try optimizer.attach(weight.*);
                try optimizer.attach(bias.*);
            }
        }

        pub fn setRequiresGrad(self: *Self, requires_grad: bool) void {
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

        fn makeWeight(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: *zg.Graph,
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
