//! Precomputed Freivalds verification.
//!
//! Key insight: r_j^T (W_j x) = (r_j^T W_j) x = v_j · x
//!
//! Keygen: compute v_j^(i) = r_j^T W_j^(i) once per matrix per layer.
//! Verify:  check v_j^(i) · x == r_j · z where z is the claimed output.
//!
//! Security: the random vectors r_j are verifier-secret. The prover sends
//! full output vectors z; the verifier computes r · z locally. False-accept
//! probability ≤ 1/p per check.

const std = @import("std");
const field = @import("field.zig");
const Fp = field.Fp;
const Fp64 = field.Fp64;
const Fp128 = field.Fp128;
const Sha256 = std.crypto.hash.sha2.Sha256;

// ═══════════════════════════════════════════════════════════════════════
// Fp variants
// ═══════════════════════════════════════════════════════════════════════

/// Precompute v = r^T W in F_p.
///
/// W is row-major (rows × cols). r has length `rows`.
/// Returns v of length `cols`. Caller owns returned memory.
///
/// Overflow budget: each |term| ≤ (P-1) × 128 < 2^39.
/// u128 accumulator overflows at 2^89 terms — all realistic models are ≤ 2^16 rows.
pub fn precomputeV(allocator: std.mem.Allocator, r: []const Fp, weight: []const i8, rows: usize, cols: usize) ![]Fp {
    std.debug.assert(r.len == rows);
    std.debug.assert(weight.len == rows * cols);

    const v = try allocator.alloc(Fp, cols);
    errdefer allocator.free(v);

    for (0..cols) |col| {
        var acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (0..rows) |row| {
            const w_val: i16 = @intCast(weight[row * cols + col]);
            const r_val: u128 = @as(u128, r[row].val);
            if (w_val >= 0) {
                acc += r_val * @as(u128, @intCast(w_val));
            } else {
                neg_acc += r_val * @as(u128, @intCast(-w_val));
            }
        }
        const pos: u64 = @intCast(acc % @as(u128, field.P));
        const neg: u64 = @intCast(neg_acc % @as(u128, field.P));
        v[col] = if (pos >= neg)
            Fp{ .val = @intCast(pos - neg) }
        else
            Fp{ .val = @intCast(pos + field.P - neg) };
    }
    return v;
}

/// Verify: does v · x == r · z?
pub fn check(v: []const Fp, x: []const i8, r: []const Fp, z: []const i32) bool {
    return Fp.dotFpI8(v, x).eql(Fp.dotFpI32(r, z));
}

// ═══════════════════════════════════════════════════════════════════════
// Fp64 variants
// ═══════════════════════════════════════════════════════════════════════

/// Overflow budget: each |term| ≤ (P64-1) × 128 < 2^68.
/// u128 overflows at 2^60 terms — all realistic models are ≤ 2^16 rows.
pub fn precomputeV64(allocator: std.mem.Allocator, r: []const Fp64, weight: []const i8, rows: usize, cols: usize) ![]Fp64 {
    std.debug.assert(r.len == rows);
    std.debug.assert(weight.len == rows * cols);

    const v = try allocator.alloc(Fp64, cols);
    errdefer allocator.free(v);

    for (0..cols) |col| {
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (0..rows) |row| {
            const w_val: i16 = @intCast(weight[row * cols + col]);
            const r_val: u128 = @as(u128, r[row].val);
            if (w_val >= 0) {
                pos_acc += r_val * @as(u128, @intCast(w_val));
            } else {
                neg_acc += r_val * @as(u128, @intCast(-w_val));
            }
        }
        const pos = Fp64.reduce(pos_acc);
        const neg = Fp64.reduce(neg_acc);
        v[col] = if (pos >= neg)
            Fp64{ .val = pos - neg }
        else
            Fp64{ .val = pos + field.P64 - neg };
    }
    return v;
}

pub fn check64(v: []const Fp64, x: []const i8, r: []const Fp64, z: []const i32) bool {
    return Fp64.dotFpI8(v, x).eql(Fp64.dotFpI32(r, z));
}

// ═══════════════════════════════════════════════════════════════════════
// Fp128 variants
// ═══════════════════════════════════════════════════════════════════════

pub fn precomputeV128(allocator: std.mem.Allocator, r: []const Fp128, weight: []const i8, rows: usize, cols: usize) ![]Fp128 {
    std.debug.assert(r.len == rows);
    std.debug.assert(weight.len == rows * cols);

    const v = try allocator.alloc(Fp128, cols);
    errdefer allocator.free(v);

    for (0..cols) |col| {
        var acc = Fp128.ZERO;
        for (0..rows) |row| {
            const w_fp = Fp128.fromI8(weight[row * cols + col]);
            acc = acc.add(r[row].mul(w_fp));
        }
        v[col] = acc;
    }
    return v;
}

pub fn check128(v: []const Fp128, x: []const i8, r: []const Fp128, z: []const i32) bool {
    return Fp128.dotFpI8(v, x).eql(Fp128.dotFpI32(r, z));
}

// ═══════════════════════════════════════════════════════════════════════
// Q8_0 block-aware Freivalds
// ═══════════════════════════════════════════════════════════════════════

const Q8_0_BLOCK_SIZE = @import("constants.zig").Q8_0_BLOCK_SIZE;

/// Derive secret batching coefficients for block Freivalds from verifier key seed.
///
/// Coefficients are derived from the verifier's secret key seed (never shared
/// with the prover), layer index, and matrix type. Strictly stronger than
/// Fiat-Shamir: zero information leakage about coefficients.
pub fn deriveBlockCoefficients(allocator: std.mem.Allocator, key_seed: *const [32]u8, layer: usize, matrix_idx: usize, n_blocks: usize) ![]Fp {
    const coeffs = try allocator.alloc(Fp, n_blocks);
    errdefer allocator.free(coeffs);

    for (0..n_blocks) |b| {
        var hasher = Sha256.init(.{});
        hasher.update("vi-block-coeff-v2");
        hasher.update(key_seed);
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(layer))));
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(matrix_idx))));
        hasher.update(&std.mem.toBytes(@as(u32, @intCast(b))));
        const hash = hasher.finalResult();
        const val = std.mem.readInt(u32, hash[0..4], .little);
        coeffs[b] = Fp.new(val);
    }
    return coeffs;
}

/// Phase A: Batched block Freivalds check for Q8_0.
///
///   Σ_b c_b · dot(v_b, x_b) == r · (Σ_b c_b · sumi_b)
pub fn checkQ8Blocks(
    v: []const Fp,
    x: []const i8,
    r: []const Fp,
    sumi: []const []const i32,
    c: []const Fp,
) bool {
    const n_blocks = sumi.len;
    std.debug.assert(c.len == n_blocks);
    std.debug.assert(v.len == n_blocks * Q8_0_BLOCK_SIZE);
    std.debug.assert(x.len == n_blocks * Q8_0_BLOCK_SIZE);

    const output_dim = r.len;

    // LHS: Σ_b c_b · dot(v_b, x_b)
    var lhs = Fp.ZERO;
    for (0..n_blocks) |b| {
        const start = b * Q8_0_BLOCK_SIZE;
        const end = start + Q8_0_BLOCK_SIZE;
        const dot_b = Fp.dotFpI8(v[start..end], x[start..end]);
        lhs = lhs.add(c[b].mul(dot_b));
    }

    // RHS: r · z' where z'[row] = Σ_b c_b · sumi_b[row]
    // Compute z' in Fp, then dot with r.
    for (sumi) |s| std.debug.assert(s.len == output_dim);
    var rhs_acc: u128 = 0;
    for (0..output_dim) |row| {
        var z_prime = Fp.ZERO;
        for (0..n_blocks) |b| {
            z_prime = z_prime.add(c[b].mul(Fp.fromI32(sumi[b][row])));
        }
        rhs_acc += @as(u128, r[row].val) * @as(u128, z_prime.val);
    }
    const rhs = Fp{ .val = @intCast(rhs_acc % @as(u128, field.P)) };

    return lhs.eql(rhs);
}

/// Phase B: Verify f32 assembly from verified block accumulators and public scales.
///
///   claimed_output[row] == Σ_b (d_w[row * n_blocks + b] · d_x[b] · sumi_b[row])
///
/// Uses canonical left-to-right accumulation in f32.
pub fn checkQ8Assembly(
    sumi: []const []const i32,
    d_w: []const f32,
    d_x: []const f32,
    claimed_output: []const f32,
    tolerance: f32,
) bool {
    const n_blocks = sumi.len;
    if (n_blocks == 0) return claimed_output.len == 0;
    const output_dim = sumi[0].len;
    std.debug.assert(d_w.len == output_dim * n_blocks);
    std.debug.assert(d_x.len == n_blocks);
    std.debug.assert(claimed_output.len == output_dim);

    for (0..output_dim) |row| {
        var acc: f32 = 0.0;
        for (0..n_blocks) |b| {
            acc += d_w[row * n_blocks + b] * d_x[b] * @as(f32, @floatFromInt(sumi[b][row]));
        }
        const diff = @abs(acc - claimed_output[row]);
        if (diff > tolerance) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "freivalds_correct" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 83 };
    const v = try precomputeV(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(check(v, &x, &r, &z));
}

test "freivalds_wrong_output" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 84 };
    const v = try precomputeV(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(!check(v, &x, &r, &z));
}

test "freivalds_negative_weights" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ -1, 2, 3, -4 };
    const r = [_]Fp{ Fp{ .val = 5 }, Fp{ .val = 10 } };
    const x = [_]i8{ 3, 7 };
    const z = [_]i32{ 11, -19 };
    const v = try precomputeV(allocator, &r, &w, 2, 2);
    defer allocator.free(v);
    try std.testing.expect(check(v, &x, &r, &z));
}

test "freivalds_identity" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const r = [_]Fp{ Fp{ .val = 42 }, Fp{ .val = 99 }, Fp{ .val = 7 } };
    const x = [_]i8{ 10, 20, 30 };
    const z = [_]i32{ 10, 20, 30 };
    const v = try precomputeV(allocator, &r, &w, 3, 3);
    defer allocator.free(v);
    try std.testing.expect(check(v, &x, &r, &z));
}

test "freivalds_64_correct" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 83 };
    const v = try precomputeV64(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(check64(v, &x, &r, &z));
}

test "freivalds_64_wrong" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 84 };
    const v = try precomputeV64(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(!check64(v, &x, &r, &z));
}

test "freivalds_128_correct" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp128{ Fp128{ .val = 10 }, Fp128{ .val = 20 }, Fp128{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 83 };
    const v = try precomputeV128(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(check128(v, &x, &r, &z));
}

test "freivalds_128_wrong" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp128{ Fp128{ .val = 10 }, Fp128{ .val = 20 }, Fp128{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    const z = [_]i32{ 23, 53, 84 };
    const v = try precomputeV128(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(!check128(v, &x, &r, &z));
}

test "q8_assembly_correct" {
    const sumi = [_][]const i32{
        &[_]i32{ 100, 200, 300 },
        &[_]i32{ 400, 500, 600 },
    };
    const d_w = [_]f32{ 1.0, 2.0, 0.5, 1.5, 1.0, 1.0 };
    const d_x = [_]f32{ 1.0, 0.5 };
    const output = [_]f32{ 500.0, 475.0, 600.0 };
    try std.testing.expect(checkQ8Assembly(&sumi, &d_w, &d_x, &output, 0.0));
}

test "q8_assembly_wrong" {
    const sumi = [_][]const i32{
        &[_]i32{ 100, 200 },
        &[_]i32{ 300, 400 },
    };
    const d_w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    const d_x = [_]f32{ 1.0, 1.0 };
    const wrong_output = [_]f32{ 401.0, 600.0 };
    try std.testing.expect(!checkQ8Assembly(&sumi, &d_w, &d_x, &wrong_output, 0.0));
    // But passes with tolerance
    try std.testing.expect(checkQ8Assembly(&sumi, &d_w, &d_x, &wrong_output, 1.5));
}
