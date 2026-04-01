//! SiLU (Swish) verification for INT8 paths.
//!
//! In the toy/unit-scale path, the gate input is first clamped to INT8:
//!   h[i] = requant(SiLU(dequant(g_i8[i])) * dequant(u_i8[i]))
//!
//! Since g_i8 has only 256 possible values, this path admits a 256-entry LUT.

const std = @import("std");
const requantize_mod = @import("requantize.zig");

/// Raw SiLU function: x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// Build the SiLU LUT for a given quantization scale.
/// Maps each INT8 value g in -128..127 to SiLU(g * scale).
pub fn buildSiluLut(scale: f32) [256]f32 {
    var lut: [256]f32 = undefined;
    for (0..256) |i| {
        const g: i8 = @bitCast(@as(u8, @intCast(i)));
        const x: f32 = @as(f32, @floatFromInt(g)) * scale;
        lut[i] = silu(x);
    }
    return lut;
}

/// Compute h = SiLU(requant(g)) * requant(u), requantized to i8.
///
/// Takes i32 accumulators from the gate and up projections. Requantizes
/// them to i8 (clamp), looks up SiLU from a 256-entry LUT, multiplies
/// by the up value, and clamps the result to i8.
///
/// Uses unit quantization scale (scale=1.0). Canonical for the toy model.
pub fn computeHUnitScale(allocator: std.mem.Allocator, g_acc: []const i32, u_acc: []const i32) ![]i8 {
    std.debug.assert(g_acc.len == u_acc.len);
    const lut = buildSiluLut(1.0);
    const result = try allocator.alloc(i8, g_acc.len);
    for (g_acc, u_acc, 0..) |g, u, i| {
        const g_i8 = requantize_mod.requantize(g);
        const u_i8 = requantize_mod.requantize(u);
        const silu_g = lut[@as(u8, @bitCast(g_i8))];
        const product = silu_g * @as(f32, @floatFromInt(u_i8));
        const rounded: i32 = @intFromFloat(@round(product));
        result[i] = @intCast(std.math.clamp(rounded, -128, 127));
    }
    return result;
}

/// Verify the SiLU + elementwise multiply step.
/// Returns true if all elements match within requantization tolerance (+-1).
pub fn checkSilu(
    g: []const i8,
    u: []const i8,
    h: []const i8,
    lut: []const f32,
    u_scale: f32,
    h_scale: f32,
    h_zero: i8,
) bool {
    std.debug.assert(g.len == u.len);
    std.debug.assert(g.len == h.len);

    for (g, u, h) |gi, ui, hi| {
        const g_idx = @as(u8, @bitCast(gi));
        const silu_g = lut[g_idx];
        const u_val: f32 = @as(f32, @floatFromInt(ui)) * u_scale;
        const product = silu_g * u_val;
        const expected_raw: i32 = @as(i32, @intFromFloat(@round(product / h_scale))) + @as(i32, h_zero);
        const expected: i8 = @intCast(std.math.clamp(expected_raw, -128, 127));
        const diff = @abs(@as(i32, hi) - @as(i32, expected));
        if (diff > 1) return false;
    }
    return true;
}

test "silu_values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), silu(1.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1423), silu(-3.0), 0.001);
}

test "build_lut" {
    const lut = buildSiluLut(0.1);
    // g=0 -> SiLU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lut[0], 1e-6);
    // g=10 -> SiLU(1.0) ≈ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), lut[10], 0.001);
}

test "compute_h_unit_scale" {
    const allocator = std.testing.allocator;
    const g = [_]i32{ 10, -5, 127, -128, 200 };
    const u = [_]i32{ 20, 30, -10, 50, -60 };
    const result = try computeHUnitScale(allocator, &g, &u);
    defer allocator.free(result);
    // Just verify it runs and produces i8 values
    for (result) |v| {
        try std.testing.expect(v >= -128 and v <= 127);
    }
}
