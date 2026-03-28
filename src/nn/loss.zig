// TODO: these ops to be migrated into the new device API design
const std = @import("std");
const builtin = @import("builtin");
const debug: bool = (builtin.mode == .Debug);

const zg = @import("../zigrad.zig");
const DeviceReference = zg.DeviceReference;
const ReduceType = zg.ReduceType;
const Shape = zg.Shape;
const NDArray = zg.NDArray;
const settings = zg.settings;
const NDTensor = zg.NDTensor;
const Graph = zg.Graph;
const Node = Graph.Node;

fn copyDeviceSliceToHostOwned(comptime T: type, device: DeviceReference, src: []const T, allocator: std.mem.Allocator) ![]T {
    const host = try allocator.alloc(T, src.len);
    errdefer allocator.free(host);

    if (device.is_host()) {
        @memcpy(host, src);
        return host;
    }

    device.sync();
    device.mem_transfer(T, src, host, .DtoH);
    device.sync();
    return host;
}

fn writeHostSliceToDevice(comptime T: type, device: DeviceReference, dst: []T, src: []const T) void {
    std.debug.assert(dst.len == src.len);

    if (device.is_host()) {
        @memcpy(dst, src);
        return;
    }

    device.mem_transfer(T, src, dst, .HtoD);
    device.sync();
}

fn softmaxForwardHostBuffer(comptime T: type, input: []const T, shape: Shape, dim: usize, output: []T) !void {
    if (dim >= shape.len) return error.InvalidDimension;
    std.debug.assert(input.len == output.len);

    @memcpy(output, input);

    const dim_size = shape.get(dim);
    const total_size = shape.size();
    const outer_size = @divExact(total_size, dim_size);
    const strides = shape.strides();

    var outer_idx: usize = 0;
    while (outer_idx < outer_size) : (outer_idx += 1) {
        const base_idx = (outer_idx / strides.get(dim)) * (strides.get(dim) * dim_size) + (outer_idx % strides.get(dim));

        var max_val = output[base_idx];
        for (1..dim_size) |j| {
            const idx = base_idx + j * strides.get(dim);
            max_val = @max(max_val, output[idx]);
        }

        var sum_exp: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * strides.get(dim);
            sum_exp += @exp(output[idx] - max_val);
        }
        const log_sum_exp = max_val + @log(sum_exp);

        for (0..dim_size) |j| {
            const idx = base_idx + j * strides.get(dim);
            output[idx] = @exp(output[idx] - log_sum_exp);
        }
    }
}

fn softmaxBackwardHostBuffer(comptime T: type, y: []const T, y_grad: []const T, x_grad: []T, shape: Shape, dim: usize) !void {
    if (dim >= shape.len) return error.InvalidDimension;
    std.debug.assert(y.len == y_grad.len);
    std.debug.assert(y.len == x_grad.len);

    const dim_size = shape.get(dim);
    const total_size = shape.size();
    const outer_size = @divExact(total_size, dim_size);
    const strides = shape.strides();

    var outer_idx: usize = 0;
    while (outer_idx < outer_size) : (outer_idx += 1) {
        const base_idx = (outer_idx / strides.get(dim)) * (strides.get(dim) * dim_size) + (outer_idx % strides.get(dim));

        var sum_grad: T = 0;
        for (0..dim_size) |j| {
            const idx = base_idx + j * strides.get(dim);
            sum_grad += y[idx] * y_grad[idx];
        }

        for (0..dim_size) |j| {
            const idx = base_idx + j * strides.get(dim);
            x_grad[idx] += y[idx] * (y_grad[idx] - sum_grad);
        }
    }
}

/// NLL entry point
/// NOTE: Reductions return NaN when "reduce" is set to false.
/// This is for c-compatibility and this value is instead traded for null.
pub fn nll(T: type, comptime config: NLLConfig) NLLType(T, config) {
    return .{};
}

pub const NLLConfig = struct {
    /// Dimensions of input tensor
    dimensions: usize,

    /// NLL expecting logits - if true, softmax will be used on input
    input_logits: bool,

    /// Specifies that the target type will be provided as an index
    target_type: enum { indices, encoding },

    /// Specifies the reduce type used
    reduce_type: ReduceType,
};

/// `NLLIndex` type from config
pub fn NLLType(T: type, comptime config: NLLConfig) type {
    return if (config.target_type == .indices) NLLIndex(T, config) else @compileError("Unimplemented");
}

fn NLLEncode(T: type, comptime config: NLLConfig) type {
    return switch (config.dimensions) {
        1 => struct {
            pub fn score(src: *NDTensor(T), trg: *const NDTensor(T), reduce: bool) !*NDTensor {
                const out = try NDTensor(T).empty(&.{1}, src.device);
                src.device.nn.nll_loss_1d_encode_forward(
                    T,
                    src.get_data(),
                    trg.get_data(),
                    out.get_data(),
                    config.input_logits,
                    reduce,
                    config.reduce_type,
                );
                return out;
            }

            //fn backward(src: *zg.NDTensor(T)) anyerror!void {
            //    const trg: usize = @intFromPtr(src._backward_ctx.?) -| 1;
            //    src.device.nn.nll_loss_1d_index_backward(T, src.get_data(), src.grad.?.data, trg, config.reduce_type);
            //}
        },
        else => @compileError("Unsupported Dimensions for NLLIndex"),
    };
}

fn NLLIndex(T: type, comptime config: NLLConfig) type {
    return switch (config.dimensions) {
        1 => struct {
            pub fn score(src: *NDTensor(T), trg_index: usize, reduce: bool) !*NDTensor {
                const out = try NDTensor(T).empty(&.{1}, src.device);
                src.device.nn.nll_loss_1d_index_forward(
                    T,
                    src.get_data(),
                    trg_index,
                    out.get_data(),
                    config.input_logits,
                    reduce,
                    config.reduce_type,
                );
            }

            //fn backward(src: *zg.NDTensor(T)) anyerror!void {
            //    const trg: usize = @intFromPtr(src._backward_ctx.?) -| 1;
            //    src.device.nn.nll_loss_1d_index_backward(T, src.get_data(), src.grad.?.data, trg, config.reduce_type);
            //}
        },
        else => @compileError("Unsupported Dimensions for NLLIndex"),
    };
}

/// Direct Mean Squared Error loss.
pub fn mse_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T)) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    const n = @as(T, @floatFromInt(y.get_size()));
    var sum_sq_diff: T = 0;
    if (y_pred.device.is_host()) {
        for (y_pred.get_data(), y.get_data()) |pred, target| {
            const diff = pred - target;
            sum_sq_diff += diff * diff;
        }
    } else {
        const allocator = std.heap.smp_allocator;
        const pred_host = try y_pred.to_host_owned(allocator);
        defer allocator.free(pred_host);
        const target_host = try y.to_host_owned(allocator);
        defer allocator.free(target_host);

        for (pred_host, target_host) |pred, target| {
            const diff = pred - target;
            sum_sq_diff += diff * diff;
        }
    }
    const mse = sum_sq_diff / n;

    const MseBwd = struct {
        pub fn backward(_: *Tensor, children: *Node.Children) !void {
            const _y_pred = children.get_bwd_upcast(Tensor, 0) orelse return;
            const _y = children.get_upcast(Tensor, 1);

            const _n = @as(T, @floatFromInt(_y.get_size()));
            const scale = @as(T, 2) / _n;

            const grad_data = try _y_pred.ensure_grad_data(0);
            if (_y_pred.device.is_host()) {
                for (grad_data, _y_pred.get_data(), _y.get_data()) |*grad_val, pred_val, target_val| {
                    grad_val.* += scale * (pred_val - target_val);
                }
                return;
            }

            const allocator = std.heap.smp_allocator;
            const pred_host = try _y_pred.to_host_owned(allocator);
            defer allocator.free(pred_host);
            const target_host = try _y.to_host_owned(allocator);
            defer allocator.free(target_host);
            const grad_host = try copyDeviceSliceToHostOwned(T, _y_pred.device, grad_data, allocator);
            defer allocator.free(grad_host);

            for (grad_host, pred_host, target_host) |*grad_val, pred_val, target_val| {
                grad_val.* += scale * (pred_val - target_val);
            }

            writeHostSliceToDevice(T, _y_pred.device, grad_data, grad_host);
        }
    };

    return try Tensor.create_dependent(MseBwd, .{
        .data = try NDArray(T).from_slice(&.{mse}, &.{1}, y_pred.device),
        .children = &.{ &y_pred.node, &y.node },
        .label = "mse",
        .device = y_pred.device,
        .gb = y_pred.node.gb,
        .callback = .{},
    });
}

/// Runs over last dim.
pub fn softmax_cross_entropy_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T)) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    var sum_loss: T = 0;
    const epsilon: T = 1e-7;
    if (y_pred.data.shape.len > 2) return error.NotSupported;
    const batch_size = if (y_pred.data.shape.len > 1) y_pred.data.shape.get(0) else 1;
    const last_dim = if (y_pred.data.shape.len > 1) y_pred.data.shape.len - 1 else 0;
    const sm_preds = try _softmax_fwd(T, y_pred, last_dim);
    if (debug) sm_preds.set_label("sm_preds");

    if (y_pred.device.is_host()) {
        for (sm_preds.get_data(), y.get_data()) |pred, target| {
            const safe_pred = @min(@max(pred, epsilon), 1.0 - epsilon);
            sum_loss -= target * @log(safe_pred);
        }
    } else {
        const allocator = std.heap.smp_allocator;
        const preds_host = try sm_preds.to_host_owned(allocator);
        defer allocator.free(preds_host);
        const target_host = try y.to_host_owned(allocator);
        defer allocator.free(target_host);

        for (preds_host, target_host) |pred, target| {
            const safe_pred = @min(@max(pred, epsilon), 1.0 - epsilon);
            sum_loss -= target * @log(safe_pred);
        }
    }
    const mean_loss = sum_loss / @as(T, @floatFromInt(batch_size));

    const CceBwd = struct {
        sm_preds: *Tensor,
        pub fn backward(_: *Tensor, children: *Node.Children, ctx: *@This()) !void {
            defer ctx.sm_preds.deinit();
            const preds = children.get_bwd_upcast(Tensor, 0) orelse return;
            const label = children.get_upcast(Tensor, 1);
            const bw_batch_size = if (preds.data.shape.len > 1) preds.data.shape.get(0) else 1;

            const grad_data = try preds.ensure_grad_data(0);
            if (preds.device.is_host()) {
                for (grad_data, ctx.sm_preds.get_data(), label.get_data()) |*bw_grad_val, bw_sm_val, bw_target_val| {
                    bw_grad_val.* += (bw_sm_val - bw_target_val) / @as(T, @floatFromInt(bw_batch_size));
                }
                return;
            }

            const allocator = std.heap.smp_allocator;
            const grad_host = try copyDeviceSliceToHostOwned(T, preds.device, grad_data, allocator);
            defer allocator.free(grad_host);
            const sm_host = try ctx.sm_preds.to_host_owned(allocator);
            defer allocator.free(sm_host);
            const label_host = try label.to_host_owned(allocator);
            defer allocator.free(label_host);

            for (grad_host, sm_host, label_host) |*bw_grad_val, bw_sm_val, bw_target_val| {
                bw_grad_val.* += (bw_sm_val - bw_target_val) / @as(T, @floatFromInt(bw_batch_size));
            }

            writeHostSliceToDevice(T, preds.device, grad_data, grad_host);
        }
    };

    return try Tensor.create_dependent(CceBwd, .{
        .data = try NDArray(T).from_slice(&.{mean_loss}, &.{1}, y_pred.device),
        .op = null,
        .children = &.{ &y_pred.node, &y.node },
        .label = "cross_entropy",
        .device = y_pred.device,
        .gb = y_pred.node.gb,
        .callback = .{ .sm_preds = sm_preds },
    });
}

// There are a few ways to do this. Could SIMD sum outside the loop with an NDArray method, but accum seems like a solid idea rn.
fn _softmax_fwd(T: type, input: *const NDTensor(T), dim: usize) !*NDTensor(T) {
    var result = try input.clone();
    errdefer result.deinit();

    if (input.device.is_host()) {
        try softmaxForwardHostBuffer(T, input.get_data(), input.data.shape, dim, result.get_data());
        return result;
    }

    const allocator = std.heap.smp_allocator;
    const input_host = try input.to_host_owned(allocator);
    defer allocator.free(input_host);
    const result_host = try allocator.alloc(T, input_host.len);
    defer allocator.free(result_host);

    try softmaxForwardHostBuffer(T, input_host, input.data.shape, dim, result_host);
    writeHostSliceToDevice(T, input.device, result.get_data(), result_host);
    return result;
}

pub fn softmax(T: type, input: *const NDTensor(T), dim: usize, device: DeviceReference) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    const result = try _softmax_fwd(T, input, dim);
    const SmaxBwd = struct {
        dim: usize,
        pub fn backward(y: *Tensor, children: *Node.Children, ctx: *@This()) !void {
            const bw_input = children.get_bwd_upcast(Tensor, 0) orelse return;
            const bw_grad = try bw_input.ensure_grad_data(0);

            if (y.device.is_host()) {
                try softmaxBackwardHostBuffer(T, y.get_data(), y.assume_grad_data(), bw_grad, y.data.shape, ctx.dim);
                return;
            }

            const allocator = std.heap.smp_allocator;
            const bw_grad_host = try copyDeviceSliceToHostOwned(T, y.device, bw_grad, allocator);
            defer allocator.free(bw_grad_host);
            const y_host = try y.to_host_owned(allocator);
            defer allocator.free(y_host);
            const y_grad_host = try copyDeviceSliceToHostOwned(T, y.device, y.assume_grad_data(), allocator);
            defer allocator.free(y_grad_host);

            try softmaxBackwardHostBuffer(T, y_host, y_grad_host, bw_grad_host, y.data.shape, ctx.dim);
            writeHostSliceToDevice(T, y.device, bw_grad, bw_grad_host);
        }
    };

    return try Tensor.create_dependent(SmaxBwd, .{
        .data = result.data,
        .children = &.{@constCast(&input.node)},
        .label = "softmax",
        .device = device,
        .gb = input.node.gb,
        .callback = .{ .dim = dim },
    });
}

pub fn smooth_l1_loss(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), beta: T) !*NDTensor(T) {
    const Tensor = NDTensor(T);

    std.debug.assert(y_pred.device.is_compatible(y.device));
    const n = @as(T, @floatFromInt(y.get_size()));
    var sum_loss: T = 0;

    if (y_pred.device.is_host()) {
        for (y_pred.get_data(), y.get_data()) |pred, target| {
            const diff: T = pred - target;
            const abs_diff: T = @abs(diff);
            if (abs_diff < beta) {
                sum_loss += 0.5 * (diff * diff) / beta;
            } else {
                sum_loss += abs_diff - (0.5 * beta);
            }
        }
    } else {
        const allocator = std.heap.smp_allocator;
        const pred_host = try y_pred.to_host_owned(allocator);
        defer allocator.free(pred_host);
        const target_host = try y.to_host_owned(allocator);
        defer allocator.free(target_host);

        for (pred_host, target_host) |pred, target| {
            const diff: T = pred - target;
            const abs_diff: T = @abs(diff);
            if (abs_diff < beta) {
                sum_loss += 0.5 * (diff * diff) / beta;
            } else {
                sum_loss += abs_diff - (0.5 * beta);
            }
        }
    }
    const loss = sum_loss / n;

    const Sl1LossBwd = struct {
        beta: T,
        pub fn backward(_: *Tensor, children: *Node.Children, ctx: *@This()) !void {
            const _y_pred = children.get_bwd_upcast(Tensor, 0) orelse return;
            const _y = children.get_upcast(Tensor, 1);
            const _beta = ctx.beta;

            const _n = @as(T, @floatFromInt(_y.get_size()));

            const grad_data = try _y_pred.ensure_grad_data(0);
            if (_y_pred.device.is_host()) {
                for (grad_data, _y_pred.get_data(), _y.get_data()) |*grad_val, pred_val, target_val| {
                    const diff = pred_val - target_val;
                    if (@abs(diff) < _beta) {
                        grad_val.* += diff / (_beta * _n);
                    } else {
                        grad_val.* += std.math.sign(diff) / _n;
                    }
                }
                return;
            }

            const allocator = std.heap.smp_allocator;
            const grad_host = try copyDeviceSliceToHostOwned(T, _y_pred.device, grad_data, allocator);
            defer allocator.free(grad_host);
            const pred_host = try _y_pred.to_host_owned(allocator);
            defer allocator.free(pred_host);
            const target_host = try _y.to_host_owned(allocator);
            defer allocator.free(target_host);

            for (grad_host, pred_host, target_host) |*grad_val, pred_val, target_val| {
                const diff = pred_val - target_val;
                if (@abs(diff) < _beta) {
                    grad_val.* += diff / (_beta * _n);
                } else {
                    grad_val.* += std.math.sign(diff) / _n;
                }
            }

            writeHostSliceToDevice(T, _y_pred.device, grad_data, grad_host);
        }
    };

    return Tensor.create_dependent(Sl1LossBwd, .{
        .data = try NDArray(T).from_slice(&.{loss}, &.{1}, y_pred.device),
        .children = &.{ &y_pred.node, &y.node },
        .label = "smooth_l1",
        .callback = .{ .beta = beta },
        .gb = y_pred.node.gb,
        .device = y_pred.device,
    });
}

/// Naive softmax 1D that uses on autograd.
/// This autograd variant is intended to by used as part of autograd system test and verification
/// dedicated kernels should generally be used for such operations.
pub fn ag_softmax_1d(T: type, input: *NDTensor(T)) !*NDTensor(T) {
    const max_val = try input.max();
    const exp_input = try (try input.sub(max_val)).exp();
    const sum = try exp_input.sum();
    return try exp_input.div(sum);
}

/// Naive Mean Squared Error 1D loss that uses on autograd.
/// This autograd variant is intended to by used as part of autograd system test and verification
/// dedicated kernels should generally be used for such operations.
pub fn ag_mse_1d(T: type, y_pred: *NDTensor(T), y: *NDTensor(T), device: DeviceReference) !*NDTensor(T) {
    var diff = try y_pred.sub(y);
    if (debug) try diff.set_label("diff");

    const sq_diff = try diff.pow(2);
    if (debug) try sq_diff.set_label("sq_diff");

    const sum_sq_diff = try sq_diff.sum();
    if (debug) try sum_sq_diff.set_label("sum_sq_diff");

    const coef = @as(T, @floatFromInt(y.get_size()));
    const coef_tensor = try NDTensor(T).from_slice(&.{coef}, null, true, device);
    if (debug) try coef_tensor.set_label("coef");

    const out = try sum_sq_diff.div(coef_tensor);
    if (debug) try out.set_label("mse");

    return out;
}
