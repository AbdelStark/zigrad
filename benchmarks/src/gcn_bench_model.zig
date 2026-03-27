const std = @import("std");
const zg = @import("zigrad");
const opspec = zg.opspec;
const Graph = zg.Graph;
const Node = Graph.Node;
const DeviceReference = zg.DeviceReference;
const NDTensor = zg.NDTensor;

pub fn GcnBenchmarkModel(comptime T: type) type {
    return struct {
        const Self = @This();
        const hidden_features = 16;

        conv1: GraphConvLayer(T),
        conv2: GraphConvLayer(T),

        pub fn init(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: *zg.Graph,
            in_features: usize,
            out_features: usize,
            seed: u64,
        ) !Self {
            return .{
                .conv1 = try GraphConvLayer(T).init(
                    allocator,
                    device,
                    graph,
                    in_features,
                    hidden_features,
                    seed +% 1,
                    "bench.gcn.conv1.weights",
                    "bench.gcn.conv1.bias",
                ),
                .conv2 = try GraphConvLayer(T).init(
                    allocator,
                    device,
                    graph,
                    hidden_features,
                    out_features,
                    seed +% 2,
                    "bench.gcn.conv2.weights",
                    "bench.gcn.conv2.bias",
                ),
            };
        }

        pub fn deinit(self: *Self) void {
            self.conv1.deinit();
            self.conv2.deinit();
            self.* = undefined;
        }

        pub fn forward(self: *Self, x: *NDTensor(T), edge_index: *NDTensor(usize)) !*NDTensor(T) {
            const nn = zg.nn(T);

            const h1 = try self.conv1.forward(x, edge_index);
            errdefer h1.deinit();
            try nn.relu_(h1);

            const h2 = try self.conv2.forward(h1, edge_index);
            h1.soft_deinit();
            return h2;
        }

        pub fn zeroGrad(self: *Self) void {
            self.conv1.zeroGrad();
            self.conv2.zeroGrad();
        }

        pub fn attachOptimizer(self: *Self, optimizer: zg.Optimizer) !void {
            try optimizer.attach(self.conv1.weights);
            try optimizer.attach(self.conv1.bias);
            try optimizer.attach(self.conv2.weights);
            try optimizer.attach(self.conv2.bias);
        }

        pub fn setRequiresGrad(self: *Self, requires_grad: bool) void {
            self.conv1.setRequiresGrad(requires_grad);
            self.conv2.setRequiresGrad(requires_grad);
        }
    };
}

fn GraphConvLayer(comptime T: type) type {
    return struct {
        const Self = @This();
        const Tensor = NDTensor(T);

        device: DeviceReference,
        graph: *zg.Graph,
        weights: *Tensor,
        bias: *Tensor,

        pub fn init(
            allocator: std.mem.Allocator,
            device: DeviceReference,
            graph: *zg.Graph,
            in_features: usize,
            out_features: usize,
            seed: u64,
            weight_label: []const u8,
            bias_label: []const u8,
        ) !Self {
            const weights = try makeWeight(
                allocator,
                device,
                graph,
                out_features,
                in_features,
                seed,
                weight_label,
            );
            errdefer {
                weights.release();
                weights.deinit();
            }

            const bias = try Tensor.zeros(device, &.{out_features}, .{
                .label = bias_label,
                .requires_grad = true,
                .acquired = true,
                .graph = graph,
            });

            return .{
                .device = device,
                .graph = graph,
                .weights = weights,
                .bias = bias,
            };
        }

        pub fn deinit(self: *Self) void {
            self.weights.release();
            self.weights.deinit();
            self.bias.release();
            self.bias.deinit();
        }

        pub fn zeroGrad(self: *Self) void {
            self.weights.setup_grad(0) catch {};
            self.bias.setup_grad(0) catch {};
        }

        pub fn setRequiresGrad(self: *Self, requires_grad: bool) void {
            if (requires_grad) {
                self.weights.enable_grad();
                self.bias.enable_grad();
            } else {
                self.weights.disable_grad();
                self.bias.disable_grad();
            }
        }

        pub fn forward(self: *Self, x: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            const h = try x.bmm(self.weights, .{ .trans_b = true });
            errdefer h.deinit();

            const y = try self.propagate(h, edge_index);
            errdefer y.deinit();
            h.soft_deinit();

            try y._add(self.bias);
            return y;
        }

        pub const propagate = propagateScatterGcnDegScaled;

        fn propagateScatterGcnDegScaled(self: *Self, h: *Tensor, edge_index: *NDTensor(usize)) !*Tensor {
            const node_count = h.get_dim(0);
            const feature_count = h.get_dim(1);
            std.debug.assert(edge_index.get_dim(0) == 2);
            const edge_count = edge_index.get_dim(1);

            const src_indices = edge_index.data.data.raw[0..edge_count];
            const tgt_indices = edge_index.data.data.raw[edge_count..];

            const deg_values = try self.graph.builder.allocator.alloc(T, node_count);
            defer self.graph.builder.allocator.free(deg_values);
            @memset(deg_values, 1);
            for (tgt_indices) |tgt| {
                deg_values[tgt] += 1;
            }

            const deg = try Tensor.from_slice(self.device, deg_values, &.{node_count}, .{
                .requires_grad = false,
                .graph = self.graph,
            });
            defer deg.deinit();

            const deg_norm = try deg.rsqrt();
            defer deg_norm.soft_deinit();
            if (zg.runtime.grad_enabled) deg_norm.enable_grad();

            return try scatterGcnDegScaled(h, deg_norm, src_indices, tgt_indices, node_count, feature_count, edge_count);
        }

        fn scatterGcnDegScaled(
            h: *Tensor,
            deg: *Tensor,
            src_indices: []const usize,
            tgt_indices: []const usize,
            node_count: usize,
            feature_count: usize,
            edge_count: usize,
        ) !*Tensor {
            const output = try Tensor.DataType.zeros(&.{ node_count, feature_count }, h.device);

            h.device.dispatch(opspec.scatter_gcn_deg_scaled(T){
                .dst = output.get_data(),
                .h = h.get_data(),
                .deg = deg.get_data(),
                .src_indices = src_indices,
                .tgt_indices = tgt_indices,
                .stride = feature_count,
                .n_edge = edge_count,
            });

            const Bwd = struct {
                src_indices: []const usize,
                tgt_indices: []const usize,
                feature_count: usize,
                edge_count: usize,

                pub fn backward(y: *Tensor, children: *Node.Children, ctx: *@This()) !void {
                    const h_ = children.get_bwd_upcast(Tensor, 0) orelse return;
                    const deg_ = children.get_bwd_upcast(Tensor, 1) orelse return;

                    const grad_output = y.assume_grad_data();
                    const grad_h = try h_.ensure_grad_data(0);
                    const grad_deg = try deg_.ensure_grad_data(0);

                    y.device.dispatch(opspec.scatter_gcn_deg_scaled_bwd(T){
                        .grad_output = grad_output,
                        .h = h_.get_data(),
                        .deg = deg_.get_data(),
                        .src_indices = ctx.src_indices,
                        .tgt_indices = ctx.tgt_indices,
                        .grad_h = grad_h,
                        .grad_deg = grad_deg,
                        .stride = ctx.feature_count,
                        .n_edge = ctx.edge_count,
                    });
                }
            };

            return try Tensor.create_dependent(Bwd, .{
                .data = output,
                .children = &.{ &h.node, &deg.node },
                .device = h.device,
                .gb = h.node.gb,
                .callback = .{
                    .src_indices = src_indices,
                    .tgt_indices = tgt_indices,
                    .feature_count = feature_count,
                    .edge_count = edge_count,
                },
            });
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
    };
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}
