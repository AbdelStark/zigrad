///! Impractical, unoptimized, first-pass conv impl for testing purposes. Contributions welcome!
const std = @import("std");
const zg = @import("../zigrad.zig");
const NDArray = zg.NDArray;
const DeviceReference = zg.DeviceReference;

pub const Conv2DOptions = struct {
    stride: usize = 1,
    padding: usize = 0,
    dilation: usize = 1,
};

pub fn conv2dOutputShape(
    input_shape: []const usize,
    weight_shape: []const usize,
    options: Conv2DOptions,
) ![4]usize {
    if (options.stride == 0 or options.dilation == 0) return error.InvalidConvolutionOptions;
    if (input_shape.len != 4 or weight_shape.len != 4) return error.InvalidConvolutionShape;
    if (input_shape[1] != weight_shape[1]) return error.IncompatibleChannels;

    const kernel_height = weight_shape[2];
    const kernel_width = weight_shape[3];
    if (kernel_height == 0 or kernel_width == 0) return error.InvalidConvolutionShape;
    if (kernel_height != kernel_width) return error.RectangularKernelUnsupported;

    const effective_kernel = options.dilation * (kernel_height - 1) + 1;
    const padded_height = input_shape[2] + (2 * options.padding);
    const padded_width = input_shape[3] + (2 * options.padding);
    if (padded_height < effective_kernel or padded_width < effective_kernel) {
        return error.InvalidConvolutionShape;
    }

    return .{
        input_shape[0],
        weight_shape[0],
        ((padded_height - effective_kernel) / options.stride) + 1,
        ((padded_width - effective_kernel) / options.stride) + 1,
    };
}

pub fn conv2dForwardIm2col(
    comptime T: type,
    input: NDArray(T),
    weights: NDArray(T),
    bias: ?NDArray(T),
    options: Conv2DOptions,
    device: DeviceReference,
) !NDArray(T) {
    const output_shape = try conv2dOutputShape(input.shape.slice(), weights.shape.slice(), options);
    const kernel_size = weights.shape.get(2);

    var col = try im2col(T, input, kernel_size, options.stride, options.padding, options.dilation, device);
    defer col.deinit(device);

    var weight_matrix = weights;
    weight_matrix._reshape(&.{
        weights.shape.get(0),
        weights.shape.get(1) * kernel_size * kernel_size,
    });

    var output = try weight_matrix.bmm(col, device, .{});
    errdefer output.deinit(device);

    if (bias) |bias_array| {
        if (bias_array.shape.len != 1 or bias_array.shape.get(0) != weights.shape.get(0)) {
            return error.InvalidBiasShape;
        }

        const output_channels = weights.shape.get(0);
        const spatial_size = output.shape.get(output.shape.len - 1);
        for (0..output.shape.get(0)) |batch_index| {
            for (0..output_channels) |channel_index| {
                const channel_bias = bias_array.get_data()[channel_index];
                const start = ((batch_index * output_channels) + channel_index) * spatial_size;
                const end = start + spatial_size;
                for (output.get_data()[start..end]) |*value| {
                    value.* += channel_bias;
                }
            }
        }
    }

    output._reshape(&output_shape);
    return output;
}

pub fn im2col(comptime T: type, input: NDArray(T), kernel_size: usize, stride: usize, padding: usize, dilation: usize, device: DeviceReference) !NDArray(T) {
    const batch_size = input.shape.get(0);
    const channels = input.shape.get(1);
    const height = input.shape.get(2);
    const width = input.shape.get(3);

    const output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    var col = try NDArray(T).empty(&[_]usize{ batch_size, col_height, col_width }, device);

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = @as(i64, @intCast(c % kernel_size));
            const h_offset = @as(i64, @intCast((c / kernel_size) % kernel_size));
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = @as(i64, @intCast(h * stride)) - @as(i64, @intCast(padding)) + h_offset * @as(i64, @intCast(dilation));
                    const w_pad = @as(i64, @intCast(w * stride)) - @as(i64, @intCast(padding)) + w_offset * @as(i64, @intCast(dilation));
                    if (h_pad >= 0 and h_pad < @as(i64, @intCast(height)) and w_pad >= 0 and w_pad < @as(i64, @intCast(width))) {
                        const input_index = b * channels * height * width + c_im * height * width + @as(usize, @intCast(h_pad)) * width + @as(usize, @intCast(w_pad));
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        col.data.raw[col_index] = input.data.raw[input_index];
                    } else {
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        col.data.raw[col_index] = 0;
                    }
                }
            }
        }
    }

    return col;
}

pub fn col2im(comptime T: type, col: NDArray(T), input_shape: []const usize, kernel_size: usize, stride: usize, padding: usize, dilation: usize, device: DeviceReference) !NDArray(T) {
    const batch_size = input_shape[0];
    const channels = input_shape[1];
    const height = input_shape[2];
    const width = input_shape[3];

    const output_height = (height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    const output_width = (width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    var im = try NDArray(T).empty(input_shape, device);
    @memset(im.get_data(), 0);

    const col_height = channels * kernel_size * kernel_size;
    const col_width = output_height * output_width;

    for (0..batch_size) |b| {
        var c: usize = 0;
        while (c < channels * kernel_size * kernel_size) : (c += 1) {
            const w_offset = @as(i64, @intCast(c % kernel_size));
            const h_offset = @as(i64, @intCast((c / kernel_size) % kernel_size));
            const c_im = c / (kernel_size * kernel_size);
            for (0..output_height) |h| {
                for (0..output_width) |w| {
                    const h_pad = @as(i64, @intCast(h * stride)) - @as(i64, @intCast(padding)) + h_offset * @as(i64, @intCast(dilation));
                    const w_pad = @as(i64, @intCast(w * stride)) - @as(i64, @intCast(padding)) + w_offset * @as(i64, @intCast(dilation));
                    if (h_pad >= 0 and h_pad < @as(i64, @intCast(height)) and w_pad >= 0 and w_pad < @as(i64, @intCast(width))) {
                        const col_index = b * col_height * col_width + c * col_width + h * output_width + w;
                        const im_index = b * channels * height * width + c_im * height * width + @as(usize, @intCast(h_pad)) * width + @as(usize, @intCast(w_pad));
                        im.data.raw[im_index] += col.data.raw[col_index];
                    }
                }
            }
        }
    }

    return im;
}

const TestOpts: zg.device.HostDevice.Options = .{
    .large_pool_size = zg.constants.@"1MiB",
    .small_pool_size = zg.constants.@"1MiB",
};

test "im2col col2im" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const input_shape = [_]usize{ 1, 1, 4, 4 };
    const kernel_size: usize = 3;
    const stride: usize = 1;
    const padding: usize = 1;
    const dilation: usize = 1;

    var input = try NDArray(f32).from_slice(&input_data, &input_shape, device);
    defer input.deinit(device);

    // im2col
    var col = try im2col(f32, input, kernel_size, stride, padding, dilation, device);
    defer col.deinit(device);

    const expected_col_data = [_]f32{
        0, 0, 0, 0, 0,  1,  2,  3,  0,  5,  6,  7,  0,  9,  10, 11,
        0, 0, 0, 0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        0, 0, 0, 0, 2,  3,  4,  0,  6,  7,  8,  0,  10, 11, 12, 0,
        0, 1, 2, 3, 0,  5,  6,  7,  0,  9,  10, 11, 0,  13, 14, 15,
        1, 2, 3, 4, 5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
        2, 3, 4, 0, 6,  7,  8,  0,  10, 11, 12, 0,  14, 15, 16, 0,
        0, 5, 6, 7, 0,  9,  10, 11, 0,  13, 14, 15, 0,  0,  0,  0,
        5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15, 16, 0,  0,  0,  0,
        6, 7, 8, 0, 10, 11, 12, 0,  14, 15, 16, 0,  0,  0,  0,  0,
    };

    try std.testing.expectEqualSlices(f32, &expected_col_data, col.get_data());

    // col2im
    var im = try col2im(f32, col, &input_shape, kernel_size, stride, padding, dilation, device);
    defer im.deinit(device);

    const exp_im_data = [_]f32{ 4, 12, 18, 16, 30, 54, 63, 48, 54, 90, 99, 72, 52, 84, 90, 64 };
    try std.testing.expectEqualSlices(f32, &exp_im_data, im.get_data());
}

test "conv2d forward im2col matches expected output with bias" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    const input_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    var input = try NDArray(f32).from_slice(&input_data, &.{ 1, 1, 3, 3 }, device);
    defer input.deinit(device);

    const weight_data = [_]f32{ 1, 0, 0, 1 };
    var weights = try NDArray(f32).from_slice(&weight_data, &.{ 1, 1, 2, 2 }, device);
    defer weights.deinit(device);

    const bias_data = [_]f32{0.5};
    var bias = try NDArray(f32).from_slice(&bias_data, &.{1}, device);
    defer bias.deinit(device);

    var output = try conv2dForwardIm2col(f32, input, weights, bias, .{}, device);
    defer output.deinit(device);

    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 2, 2 }, output.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{ 6.5, 8.5, 12.5, 14.5 }, output.get_data());
}

test "conv2d bias applies per output channel across spatial positions" {
    var cpu = zg.device.HostDevice.init_advanced(TestOpts);
    defer cpu.deinit();
    const device = cpu.reference();

    const input_data = [_]f32{
        1,  2,  3,  4,
        5,  6,  7,  8,
        9,  10, 11, 12,
        13, 14, 15, 16,
    };
    var input = try NDArray(f32).from_slice(&input_data, &.{ 1, 1, 4, 4 }, device);
    defer input.deinit(device);

    const weight_data = [_]f32{
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
    var weights = try NDArray(f32).from_slice(&weight_data, &.{ 2, 1, 2, 2 }, device);
    defer weights.deinit(device);

    const bias_data = [_]f32{ 0.25, -0.5 };
    var bias = try NDArray(f32).from_slice(&bias_data, &.{2}, device);
    defer bias.deinit(device);

    var output = try conv2dForwardIm2col(f32, input, weights, bias, .{}, device);
    defer output.deinit(device);

    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3, 3 }, output.shape.slice());
    try std.testing.expectEqualSlices(f32, &.{
        0.25, 0.25, 0.25,
        0.25, 0.25, 0.25,
        0.25, 0.25, 0.25,
        -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5,
    }, output.get_data());
}

test "conv2d output shape validates supported kernels" {
    const valid = try conv2dOutputShape(
        &.{ 2, 3, 8, 8 },
        &.{ 4, 3, 3, 3 },
        .{ .stride = 2, .padding = 1, .dilation = 1 },
    );
    try std.testing.expectEqualSlices(usize, &.{ 2, 4, 4, 4 }, valid[0..]);

    try std.testing.expectError(
        error.RectangularKernelUnsupported,
        conv2dOutputShape(&.{ 1, 1, 4, 4 }, &.{ 1, 1, 2, 3 }, .{}),
    );
    try std.testing.expectError(
        error.IncompatibleChannels,
        conv2dOutputShape(&.{ 1, 2, 4, 4 }, &.{ 1, 1, 3, 3 }, .{}),
    );
}
