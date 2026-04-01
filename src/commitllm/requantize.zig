//! INT8 requantization: clamp i32 accumulators to [-128, 127].
//!
//! Toy-model only. Production W8A8 models use the full bridge chain
//! (dequant -> residual -> RMSNorm -> quantize).

const std = @import("std");

/// Clamp i32 accumulator to i8 range [-128, 127].
pub fn requantize(acc: i32) i8 {
    return @intCast(std.math.clamp(acc, -128, 127));
}

/// Requantize a slice of i32 accumulators to i8. Caller owns returned slice.
pub fn requantizeSlice(allocator: std.mem.Allocator, acc: []const i32) ![]i8 {
    const result = try allocator.alloc(i8, acc.len);
    for (acc, 0..) |v, i| {
        result[i] = requantize(v);
    }
    return result;
}

/// Scale-aware requantization (toy / last-layer fallback only).
///
/// output = round(acc * scale_w * scale_x / scale_out).clamp(-128, 127)
///
/// For native INT8 (scale_w == 0.0): falls back to toy-model clamp.
pub fn bridgeRequantize(acc: i32, scale_w: f32, scale_x: f32, scale_out: f32) i8 {
    if (scale_w == 0.0) return requantize(acc);
    const combined: f64 = @as(f64, scale_w) * @as(f64, scale_x) / @as(f64, scale_out);
    const val: f64 = @as(f64, @as(f32, @floatFromInt(acc))) * combined;
    const rounded: i32 = @intFromFloat(@round(val));
    return @intCast(std.math.clamp(rounded, -128, 127));
}

test "requantize_clamp" {
    try std.testing.expectEqual(@as(i8, 50), requantize(50));
    try std.testing.expectEqual(@as(i8, 127), requantize(200));
    try std.testing.expectEqual(@as(i8, -128), requantize(-200));
    try std.testing.expectEqual(@as(i8, 127), requantize(127));
    try std.testing.expectEqual(@as(i8, -128), requantize(-128));
}

test "bridge_requantize_fallback" {
    // scale_w == 0.0 -> same as requantize (clamp)
    try std.testing.expectEqual(@as(i8, 50), bridgeRequantize(50, 0.0, 1.0, 1.0));
    try std.testing.expectEqual(@as(i8, 127), bridgeRequantize(300, 0.0, 1.0, 1.0));
}

test "bridge_requantize_identity" {
    try std.testing.expectEqual(@as(i8, 10), bridgeRequantize(10, 1.0, 1.0, 1.0));
    try std.testing.expectEqual(@as(i8, -20), bridgeRequantize(-20, 1.0, 1.0, 1.0));
}

test "bridge_requantize_scaling" {
    // acc=1000, scale_w=0.01, scale_x=0.5, scale_out=0.1
    // output = round(1000 * 0.01 * 0.5 / 0.1) = round(50) = 50
    try std.testing.expectEqual(@as(i8, 50), bridgeRequantize(1000, 0.01, 0.5, 0.1));
}
