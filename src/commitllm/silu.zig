//! SiLU (Swish) verification for both toy and production INT8 paths.
//!
//! Toy path: gate input clamped to INT8, 256-entry LUT.
//! Production W8A8 path: opened i32 accumulators with recorded scales.

const std = @import("std");
const requantize_mod = @import("requantize.zig");

/// SiLU(x) = x × σ(x) = x / (1 + exp(-x)), in f32.
pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

/// SiLU in f64 for production W8A8 bridge verification.
pub fn siluF64(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

/// Build the SiLU LUT for a given quantization scale.
/// Maps each INT8 value g in -128..127 to SiLU(g × scale).
pub fn buildSiluLut(scale: f32) [256]f32 {
    var lut: [256]f32 = undefined;
    for (0..256) |i| {
        const g: i8 = @bitCast(@as(u8, @intCast(i)));
        const x: f32 = @as(f32, @floatFromInt(g)) * scale;
        lut[i] = silu(x);
    }
    return lut;
}

/// Compute h = SiLU(requant(g)) × requant(u), requantized to i8.
///
/// Unit quantization scale (scale=1.0). Canonical for the toy model.
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

/// Scale-aware SiLU bridge for W8A8 models.
///
///   g_f = g_i32 × scale_w_g × scale_x_ffn
///   u_f = u_i32 × scale_w_u × scale_x_ffn
///   h_f = SiLU(g_f) × u_f
///   h_i8 = round(h_f / scale_h).clamp(-128, 127)
///
/// Falls back to `computeHUnitScale` when scale_w_g == 0.0 (native INT8).
pub fn computeHScaled(
    allocator: std.mem.Allocator,
    g_acc: []const i32,
    u_acc: []const i32,
    scale_w_g: f32,
    scale_w_u: f32,
    scale_x_ffn: f32,
    scale_h: f32,
) ![]i8 {
    if (scale_w_g == 0.0) return computeHUnitScale(allocator, g_acc, u_acc);
    std.debug.assert(g_acc.len == u_acc.len);

    const dequant_g: f64 = @as(f64, scale_w_g) * @as(f64, scale_x_ffn);
    const dequant_u: f64 = @as(f64, scale_w_u) * @as(f64, scale_x_ffn);
    const inv_scale_h: f64 = 1.0 / @as(f64, scale_h);

    const result = try allocator.alloc(i8, g_acc.len);
    for (g_acc, u_acc, 0..) |g, u, i| {
        const g_f: f64 = @as(f64, @floatFromInt(g)) * dequant_g;
        const u_f: f64 = @as(f64, @floatFromInt(u)) * dequant_u;
        const h_f = siluF64(g_f) * u_f;
        const quantized: i32 = @intFromFloat(@round(h_f * inv_scale_h));
        result[i] = @intCast(std.math.clamp(quantized, -128, 127));
    }
    return result;
}

/// Per-channel SiLU bridge for W8A8 models with per-channel weight scales.
pub fn computeHPerChannel(
    allocator: std.mem.Allocator,
    g_acc: []const i32,
    u_acc: []const i32,
    scale_w_g: []const f32,
    scale_w_u: []const f32,
    scale_x_ffn: f32,
    scale_h: f32,
) ![]i8 {
    std.debug.assert(g_acc.len == u_acc.len);
    std.debug.assert(g_acc.len == scale_w_g.len);
    std.debug.assert(u_acc.len == scale_w_u.len);

    const sx: f64 = @as(f64, scale_x_ffn);
    const inv_scale_h: f64 = 1.0 / @as(f64, scale_h);

    const result = try allocator.alloc(i8, g_acc.len);
    for (g_acc, u_acc, scale_w_g, scale_w_u, 0..) |g, u, swg, swu, i| {
        const g_f: f64 = @as(f64, @floatFromInt(g)) * @as(f64, swg) * sx;
        const u_f: f64 = @as(f64, @floatFromInt(u)) * @as(f64, swu) * sx;
        const h_f = siluF64(g_f) * u_f;
        const quantized: i32 = @intFromFloat(@round(h_f * inv_scale_h));
        result[i] = @intCast(std.math.clamp(quantized, -128, 127));
    }
    return result;
}

/// Verify the SiLU + elementwise multiply step.
/// Returns true if all elements match within requantization tolerance (±1).
///
/// The ±1 tolerance accounts for legitimate rounding direction ambiguity
/// at exact 0.5 boundaries in the quantization step.
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

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "silu_values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), silu(1.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1423), silu(-3.0), 0.001);
}

test "silu_f64_values" {
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), siluF64(0.0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7310585786), siluF64(1.0), 1e-6);
}

test "build_lut" {
    const lut = buildSiluLut(0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), lut[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7311), lut[10], 0.001);
}

test "compute_h_unit_scale" {
    const allocator = std.testing.allocator;
    const g = [_]i32{ 10, -5, 127, -128, 200 };
    const u = [_]i32{ 20, 30, -10, 50, -60 };
    const result = try computeHUnitScale(allocator, &g, &u);
    defer allocator.free(result);
    for (result) |v| {
        try std.testing.expect(v >= -128 and v <= 127);
    }
}

test "compute_h_scaled_fallback" {
    const allocator = std.testing.allocator;
    const g = [_]i32{ 10, -5, 127, -128, 200 };
    const u = [_]i32{ 20, 30, -10, 50, -60 };
    const unit = try computeHUnitScale(allocator, &g, &u);
    defer allocator.free(unit);
    const scaled = try computeHScaled(allocator, &g, &u, 0.0, 0.0, 1.0, 1.0);
    defer allocator.free(scaled);
    try std.testing.expectEqualSlices(i8, unit, scaled);
}

test "compute_h_scaled_known" {
    const allocator = std.testing.allocator;
    // g_f = 100 * 0.01 * 0.5 = 0.5; u_f = 50 * 0.02 * 0.5 = 0.5
    // SiLU(0.5) ≈ 0.31122; h_f = 0.31122 * 0.5 = 0.15561
    // h_i8 = round(0.15561 / 0.1) = round(1.5561) = 2
    const result = try computeHScaled(allocator, &[_]i32{100}, &[_]i32{50}, 0.01, 0.02, 0.5, 0.1);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(i8, 2), result[0]);
}
