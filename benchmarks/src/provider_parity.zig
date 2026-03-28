const std = @import("std");
const zg = @import("zigrad");
const conv_utils = zg.conv_utils;
const test_support = @import("test_support.zig");

const Array = zg.NDArray(f32);
const Tensor = zg.NDTensor(f32);

const parity_abs_tolerance: f32 = 512 * std.math.floatEps(f32);
const parity_rel_tolerance: f32 = 512 * std.math.floatEps(f32);

const BmmGradients = struct {
    lhs: []f32,
    rhs: []f32,
};

fn matrixSize(shape: []const usize) usize {
    return shape[shape.len - 2] * shape[shape.len - 1];
}

fn logicalMatrixRows(shape: []const usize, trans: bool) usize {
    return if (trans) shape[shape.len - 1] else shape[shape.len - 2];
}

fn logicalMatrixCols(shape: []const usize, trans: bool) usize {
    return if (trans) shape[shape.len - 2] else shape[shape.len - 1];
}

fn alignedBatchDim(shape: []const usize, aligned_rank: usize, aligned_index: usize) usize {
    const offset = aligned_rank - shape.len;
    if (aligned_index < offset) return 1;
    return shape[aligned_index - offset];
}

fn broadcastBatchShape(
    lhs_batch_shape: []const usize,
    rhs_batch_shape: []const usize,
    out: *[8]usize,
) ![]const usize {
    const rank = @max(lhs_batch_shape.len, rhs_batch_shape.len);
    std.debug.assert(rank <= out.len);

    for (0..rank) |aligned_index| {
        const lhs_dim = alignedBatchDim(lhs_batch_shape, rank, aligned_index);
        const rhs_dim = alignedBatchDim(rhs_batch_shape, rank, aligned_index);
        if (lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1) {
            return error.IncompatibleShapes;
        }
        out[aligned_index] = @max(lhs_dim, rhs_dim);
    }

    return out[0..rank];
}

fn encodeRowMajor(indices: []const usize, shape: []const usize) usize {
    var linear: usize = 0;
    for (shape, 0..) |dim, index| {
        linear = (linear * dim) + indices[index];
    }
    return linear;
}

fn decodeRowMajor(shape: []const usize, linear_index: usize, out: []usize) void {
    std.debug.assert(shape.len == out.len);

    var remaining = linear_index;
    var index = shape.len;
    while (index > 0) : (index -= 1) {
        const dim = shape[index - 1];
        out[index - 1] = if (dim == 0) 0 else remaining % dim;
        if (dim != 0) remaining /= dim;
    }
}

fn batchLinearIndex(
    output_batch_shape: []const usize,
    operand_batch_shape: []const usize,
    output_batch_indices: []const usize,
) usize {
    if (operand_batch_shape.len == 0) return 0;

    var operand_indices: [8]usize = undefined;
    const offset = output_batch_shape.len - operand_batch_shape.len;
    for (operand_batch_shape, 0..) |dim, index| {
        operand_indices[index] = if (dim == 1) 0 else output_batch_indices[offset + index];
    }
    return encodeRowMajor(operand_indices[0..operand_batch_shape.len], operand_batch_shape);
}

fn operandValue(
    data: []const f32,
    base: usize,
    original_cols: usize,
    row: usize,
    col: usize,
    trans: bool,
) f32 {
    const source_row = if (trans) col else row;
    const source_col = if (trans) row else col;
    return data[base + (source_row * original_cols) + source_col];
}

fn addOperandGradient(
    grad: []f32,
    base: usize,
    original_cols: usize,
    row: usize,
    col: usize,
    trans: bool,
    delta: f32,
) void {
    const source_row = if (trans) col else row;
    const source_col = if (trans) row else col;
    grad[base + (source_row * original_cols) + source_col] += delta;
}

fn referenceBmmOutputShape(
    lhs_shape: []const usize,
    rhs_shape: []const usize,
    trans_a: bool,
    trans_b: bool,
    out: *[8]usize,
) ![]const usize {
    const lhs_k = logicalMatrixCols(lhs_shape, trans_a);
    const rhs_k = logicalMatrixRows(rhs_shape, trans_b);
    if (lhs_k != rhs_k) return error.IncompatibleShapes;

    const lhs_batch = lhs_shape[0 .. lhs_shape.len - 2];
    const rhs_batch = rhs_shape[0 .. rhs_shape.len - 2];
    const batch_shape = try broadcastBatchShape(lhs_batch, rhs_batch, out);

    out[batch_shape.len] = logicalMatrixRows(lhs_shape, trans_a);
    out[batch_shape.len + 1] = logicalMatrixCols(rhs_shape, trans_b);
    return out[0 .. batch_shape.len + 2];
}

fn referenceBmmForward(
    allocator: std.mem.Allocator,
    lhs_data: []const f32,
    lhs_shape: []const usize,
    rhs_data: []const f32,
    rhs_shape: []const usize,
    trans_a: bool,
    trans_b: bool,
) ![]f32 {
    var output_shape_storage: [8]usize = undefined;
    const output_shape = try referenceBmmOutputShape(
        lhs_shape,
        rhs_shape,
        trans_a,
        trans_b,
        &output_shape_storage,
    );
    const output = try allocator.alloc(f32, test_support.countElements(output_shape));

    const lhs_batch_shape = lhs_shape[0 .. lhs_shape.len - 2];
    const rhs_batch_shape = rhs_shape[0 .. rhs_shape.len - 2];
    const output_batch_shape = output_shape[0 .. output_shape.len - 2];

    const batch_count = test_support.countElements(output_batch_shape);
    const rows = output_shape[output_shape.len - 2];
    const cols = output_shape[output_shape.len - 1];
    const inner_dim = logicalMatrixCols(lhs_shape, trans_a);
    const lhs_cols = lhs_shape[lhs_shape.len - 1];
    const rhs_cols = rhs_shape[rhs_shape.len - 1];

    var batch_indices: [8]usize = undefined;
    for (0..batch_count) |batch_linear| {
        decodeRowMajor(output_batch_shape, batch_linear, batch_indices[0..output_batch_shape.len]);
        const lhs_batch_linear = batchLinearIndex(
            output_batch_shape,
            lhs_batch_shape,
            batch_indices[0..output_batch_shape.len],
        );
        const rhs_batch_linear = batchLinearIndex(
            output_batch_shape,
            rhs_batch_shape,
            batch_indices[0..output_batch_shape.len],
        );

        const lhs_base = lhs_batch_linear * matrixSize(lhs_shape);
        const rhs_base = rhs_batch_linear * matrixSize(rhs_shape);
        const output_base = batch_linear * rows * cols;

        for (0..rows) |row| {
            for (0..cols) |col| {
                var acc: f32 = 0.0;
                for (0..inner_dim) |inner| {
                    acc += operandValue(lhs_data, lhs_base, lhs_cols, row, inner, trans_a) *
                        operandValue(rhs_data, rhs_base, rhs_cols, inner, col, trans_b);
                }
                output[output_base + (row * cols) + col] = acc;
            }
        }
    }

    return output;
}

fn referenceBmmUnitGradients(
    allocator: std.mem.Allocator,
    lhs_data: []const f32,
    lhs_shape: []const usize,
    rhs_data: []const f32,
    rhs_shape: []const usize,
    trans_a: bool,
    trans_b: bool,
) !BmmGradients {
    var output_shape_storage: [8]usize = undefined;
    const output_shape = try referenceBmmOutputShape(
        lhs_shape,
        rhs_shape,
        trans_a,
        trans_b,
        &output_shape_storage,
    );

    const lhs_grad = try allocator.alloc(f32, lhs_data.len);
    errdefer allocator.free(lhs_grad);
    const rhs_grad = try allocator.alloc(f32, rhs_data.len);
    errdefer allocator.free(rhs_grad);
    @memset(lhs_grad, 0);
    @memset(rhs_grad, 0);

    const lhs_batch_shape = lhs_shape[0 .. lhs_shape.len - 2];
    const rhs_batch_shape = rhs_shape[0 .. rhs_shape.len - 2];
    const output_batch_shape = output_shape[0 .. output_shape.len - 2];

    const batch_count = test_support.countElements(output_batch_shape);
    const rows = output_shape[output_shape.len - 2];
    const cols = output_shape[output_shape.len - 1];
    const inner_dim = logicalMatrixCols(lhs_shape, trans_a);
    const lhs_cols = lhs_shape[lhs_shape.len - 1];
    const rhs_cols = rhs_shape[rhs_shape.len - 1];

    var batch_indices: [8]usize = undefined;
    for (0..batch_count) |batch_linear| {
        decodeRowMajor(output_batch_shape, batch_linear, batch_indices[0..output_batch_shape.len]);
        const lhs_batch_linear = batchLinearIndex(
            output_batch_shape,
            lhs_batch_shape,
            batch_indices[0..output_batch_shape.len],
        );
        const rhs_batch_linear = batchLinearIndex(
            output_batch_shape,
            rhs_batch_shape,
            batch_indices[0..output_batch_shape.len],
        );

        const lhs_base = lhs_batch_linear * matrixSize(lhs_shape);
        const rhs_base = rhs_batch_linear * matrixSize(rhs_shape);

        for (0..rows) |row| {
            for (0..cols) |col| {
                for (0..inner_dim) |inner| {
                    const lhs_value = operandValue(lhs_data, lhs_base, lhs_cols, row, inner, trans_a);
                    const rhs_value = operandValue(rhs_data, rhs_base, rhs_cols, inner, col, trans_b);

                    addOperandGradient(lhs_grad, lhs_base, lhs_cols, row, inner, trans_a, rhs_value);
                    addOperandGradient(rhs_grad, rhs_base, rhs_cols, inner, col, trans_b, lhs_value);
                }
            }
        }
    }

    return .{
        .lhs = lhs_grad,
        .rhs = rhs_grad,
    };
}

fn referenceMatvec(
    allocator: std.mem.Allocator,
    matrix_data: []const f32,
    matrix_shape: []const usize,
    vector_data: []const f32,
    initial_output: []const f32,
    config: Array.MatvecConfig,
) ![]f32 {
    const rows = logicalMatrixRows(matrix_shape, config.trans_a);
    const cols = logicalMatrixCols(matrix_shape, config.trans_a);
    if (cols != vector_data.len or rows != initial_output.len) return error.InvalidShape;

    const output = try allocator.alloc(f32, rows);
    const matrix_cols = matrix_shape[matrix_shape.len - 1];

    for (0..rows) |row| {
        var acc: f32 = 0.0;
        for (0..cols) |col| {
            acc += operandValue(matrix_data, 0, matrix_cols, row, col, config.trans_a) * vector_data[col];
        }
        output[row] = (config.alpha * acc) + (config.beta * initial_output[row]);
    }

    return output;
}

fn referenceConv2d(
    allocator: std.mem.Allocator,
    input_data: []const f32,
    input_shape: []const usize,
    weight_data: []const f32,
    weight_shape: []const usize,
    bias_data: ?[]const f32,
    options: conv_utils.Conv2DOptions,
) ![]f32 {
    const output_shape = try conv_utils.conv2dOutputShape(input_shape, weight_shape, options);
    const output = try allocator.alloc(f32, test_support.countElements(output_shape[0..]));

    const batch_size = input_shape[0];
    const input_channels = input_shape[1];
    const input_height = input_shape[2];
    const input_width = input_shape[3];
    const output_channels = weight_shape[0];
    const kernel_height = weight_shape[2];
    const kernel_width = weight_shape[3];
    const output_height = output_shape[2];
    const output_width = output_shape[3];

    for (0..batch_size) |batch| {
        for (0..output_channels) |output_channel| {
            for (0..output_height) |out_row| {
                for (0..output_width) |out_col| {
                    var acc: f32 = if (bias_data) |bias| bias[output_channel] else 0.0;

                    for (0..input_channels) |input_channel| {
                        for (0..kernel_height) |kernel_row| {
                            for (0..kernel_width) |kernel_col| {
                                const input_row = @as(i64, @intCast(out_row * options.stride)) -
                                    @as(i64, @intCast(options.padding)) +
                                    @as(i64, @intCast(kernel_row * options.dilation));
                                const input_col = @as(i64, @intCast(out_col * options.stride)) -
                                    @as(i64, @intCast(options.padding)) +
                                    @as(i64, @intCast(kernel_col * options.dilation));

                                if (input_row < 0 or
                                    input_row >= @as(i64, @intCast(input_height)) or
                                    input_col < 0 or
                                    input_col >= @as(i64, @intCast(input_width)))
                                {
                                    continue;
                                }

                                const input_index = (((batch * input_channels) + input_channel) * input_height + @as(usize, @intCast(input_row))) * input_width + @as(usize, @intCast(input_col));
                                const weight_index = (((output_channel * input_channels) + input_channel) * kernel_height + kernel_row) * kernel_width + kernel_col;
                                acc += input_data[input_index] * weight_data[weight_index];
                            }
                        }
                    }

                    const output_index = (((batch * output_channels) + output_channel) * output_height + out_row) * output_width + out_col;
                    output[output_index] = acc;
                }
            }
        }
    }

    return output;
}

test "host matvec parity covers alpha beta semantics" {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const matrix_shape = [_]usize{ 3, 2 };
    const vector_shape = [_]usize{2};
    const output_shape = [_]usize{3};
    const config: Array.MatvecConfig = .{
        .alpha = -0.75,
        .beta = 0.5,
    };

    const matrix_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&matrix_shape),
        101,
    );
    defer std.testing.allocator.free(matrix_values);
    const vector_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&vector_shape),
        103,
    );
    defer std.testing.allocator.free(vector_values);
    const output_init = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&output_shape),
        107,
    );
    defer std.testing.allocator.free(output_init);

    const expected = try referenceMatvec(
        std.testing.allocator,
        matrix_values,
        &matrix_shape,
        vector_values,
        output_init,
        config,
    );
    defer std.testing.allocator.free(expected);

    var matrix = try Array.from_slice(matrix_values, &matrix_shape, device);
    defer matrix.deinit(device);
    var vector = try Array.from_slice(vector_values, &vector_shape, device);
    defer vector.deinit(device);
    var output = try Array.from_slice(output_init, &output_shape, device);
    defer output.deinit(device);

    host.resetOpTelemetry();
    Array.matvec_(matrix, vector, &output, device, config);

    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected,
        output.get_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
    try std.testing.expectEqual(@as(u64, 1), host.opTelemetry().matvec_calls);
}

test "host direct batched matmul accumulation matches reference" {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_shape = [_]usize{ 2, 2, 3 };
    const rhs_shape = [_]usize{ 2, 3, 2 };
    const output_shape = [_]usize{ 2, 2, 2 };
    const config: Array.BmmConfig = .{
        .alpha = 1.25,
        .beta = -0.5,
    };

    const lhs_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&lhs_shape),
        149,
    );
    defer std.testing.allocator.free(lhs_values);
    const rhs_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&rhs_shape),
        151,
    );
    defer std.testing.allocator.free(rhs_values);
    const output_init = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&output_shape),
        157,
    );
    defer std.testing.allocator.free(output_init);

    const raw_reference = try referenceBmmForward(
        std.testing.allocator,
        lhs_values,
        &lhs_shape,
        rhs_values,
        &rhs_shape,
        config.trans_a,
        config.trans_b,
    );
    defer std.testing.allocator.free(raw_reference);

    const expected = try std.testing.allocator.alloc(f32, raw_reference.len);
    defer std.testing.allocator.free(expected);
    for (raw_reference, output_init, 0..) |reference_value, output_value, index| {
        expected[index] = (config.alpha * reference_value) + (config.beta * output_value);
    }

    var lhs = try Array.from_slice(lhs_values, &lhs_shape, device);
    defer lhs.deinit(device);
    var rhs = try Array.from_slice(rhs_values, &rhs_shape, device);
    defer rhs.deinit(device);
    var output = try Array.from_slice(output_init, &output_shape, device);
    defer output.deinit(device);

    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
    try lhs.bmm_acc_(rhs, &output, device, config);

    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected,
        output.get_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
    try std.testing.expectEqual(@as(u64, 2), host.opTelemetry().matmul_calls);
    try std.testing.expectEqual(@as(u64, 1), host.opTelemetry().bmm_acc_calls);
    try std.testing.expectEqual(@as(u64, 1), host.dispatchTelemetry().direct_bmm_dispatches);
}

test "host nested broadcast batched matmul matches reference forward and backward" {
    var graph = zg.Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const lhs_shape = [_]usize{ 2, 2, 2, 3 };
    const rhs_shape = [_]usize{ 2, 1, 3, 2 };
    const opts: zg.TensorOpts = .{
        .requires_grad = true,
        .graph = &graph,
    };

    const lhs_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&lhs_shape),
        211,
    );
    defer std.testing.allocator.free(lhs_values);
    const rhs_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&rhs_shape),
        223,
    );
    defer std.testing.allocator.free(rhs_values);

    const expected_output = try referenceBmmForward(
        std.testing.allocator,
        lhs_values,
        &lhs_shape,
        rhs_values,
        &rhs_shape,
        false,
        false,
    );
    defer std.testing.allocator.free(expected_output);

    const expected_gradients = try referenceBmmUnitGradients(
        std.testing.allocator,
        lhs_values,
        &lhs_shape,
        rhs_values,
        &rhs_shape,
        false,
        false,
    );
    defer std.testing.allocator.free(expected_gradients.lhs);
    defer std.testing.allocator.free(expected_gradients.rhs);

    const lhs = try Tensor.from_slice(device, lhs_values, &lhs_shape, opts);
    defer lhs.deinit();
    const rhs = try Tensor.from_slice(device, rhs_values, &rhs_shape, opts);
    defer rhs.deinit();

    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
    const output = try lhs.bmm(rhs, .{});
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, output.get_shape());
    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected_output,
        output.get_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
    try std.testing.expectEqual(@as(u64, 4), host.opTelemetry().matmul_calls);
    try std.testing.expectEqual(@as(u64, 1), host.dispatchTelemetry().fallback_bmm_dispatches);
    try std.testing.expectEqual(@as(u64, 4), host.dispatchTelemetry().fallback_bmm_batches);

    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
    try output.backward();

    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected_gradients.lhs,
        lhs.assume_grad_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected_gradients.rhs,
        rhs.assume_grad_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
}

test "legacy conv2d im2col lowering matches reference convolution" {
    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const input_shape = [_]usize{ 2, 2, 5, 5 };
    const weight_shape = [_]usize{ 3, 2, 3, 3 };
    const bias_shape = [_]usize{3};
    const options: conv_utils.Conv2DOptions = .{
        .padding = 1,
    };

    const input_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&input_shape),
        307,
    );
    defer std.testing.allocator.free(input_values);
    const weight_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&weight_shape),
        311,
    );
    defer std.testing.allocator.free(weight_values);
    const bias_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&bias_shape),
        313,
    );
    defer std.testing.allocator.free(bias_values);

    const expected = try referenceConv2d(
        std.testing.allocator,
        input_values,
        &input_shape,
        weight_values,
        &weight_shape,
        bias_values,
        options,
    );
    defer std.testing.allocator.free(expected);

    var input = try Array.from_slice(input_values, &input_shape, device);
    defer input.deinit(device);
    var weights = try Array.from_slice(weight_values, &weight_shape, device);
    defer weights.deinit(device);
    var bias = try Array.from_slice(bias_values, &bias_shape, device);
    defer bias.deinit(device);

    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
    var output = try conv_utils.conv2dForwardIm2col(f32, input, weights, bias, options, device);
    defer output.deinit(device);

    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 5, 5 }, output.shape.slice());
    try test_support.expectApproxEqAbsRelSlices(
        f32,
        expected,
        output.get_data(),
        parity_abs_tolerance,
        parity_rel_tolerance,
    );
    try std.testing.expectEqual(@as(u64, 2), host.opTelemetry().matmul_calls);
    try std.testing.expectEqual(@as(u64, 1), host.opTelemetry().bmm_acc_calls);
    try std.testing.expectEqual(@as(u64, 1), host.dispatchTelemetry().direct_bmm_dispatches);
}
