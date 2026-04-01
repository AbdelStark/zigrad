//! Precomputed Freivalds verification.
//!
//! Key insight: r_j^T (W_j x) = (r_j^T W_j) x = v_j . x
//!
//! Keygen (verifier-side): compute v_j^(i) = r_j^T W_j^(i) once per matrix per layer.
//! Verification: check v_j^(i) . x == r_j . z where z is the claimed output.

const std = @import("std");
const field = @import("field.zig");
const Fp = field.Fp;
const Fp64 = field.Fp64;

/// Precompute v = r^T W in F_p.
///
/// W is stored row-major: W[row * cols + col], shape (rows, cols).
/// r has length `rows` (output dimension m_j).
/// Returns v of length `cols` (input dimension n_j). Caller owns memory.
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
        const P_u128: u128 = @as(u128, field.P);
        const pos: u64 = @intCast(acc % P_u128);
        const neg: u64 = @intCast(neg_acc % P_u128);
        v[col] = if (pos >= neg)
            Fp{ .val = @intCast(pos - neg) }
        else
            Fp{ .val = @intCast(pos + field.P - neg) };
    }
    return v;
}

/// Verify a single matrix multiplication: does v . x == r . z?
///
/// v = precomputed r^T W (length = input_dim)
/// x = input vector (INT8, length = input_dim)
/// r = random vector (length = output_dim)
/// z = claimed output W*x (i32 accumulators, length = output_dim)
pub fn check(v: []const Fp, x: []const i8, r: []const Fp, z: []const i32) bool {
    const lhs = Fp.dotFpI8(v, x);
    const rhs = Fp.dotFpI32(r, z);
    return lhs.eql(rhs);
}

// ---------------------------------------------------------------------------
// Fp64 variants
// ---------------------------------------------------------------------------

/// Precompute v = r^T W in F_p (64-bit Mersenne prime).
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

/// Verify using Fp64.
pub fn check64(v: []const Fp64, x: []const i8, r: []const Fp64, z: []const i32) bool {
    const lhs = Fp64.dotFpI8(v, x);
    const rhs = Fp64.dotFpI32(r, z);
    return lhs.eql(rhs);
}

// ===========================================================================
// Tests
// ===========================================================================

test "freivalds_correct" {
    const allocator = std.testing.allocator;
    // 3x2 matrix W, r of length 3, x of length 2
    // W = [[1, 2], [3, 4], [5, 6]]
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const r = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const x = [_]i8{ 7, 8 };
    // z = W * x = [1*7+2*8, 3*7+4*8, 5*7+6*8] = [23, 53, 83]
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
    const z = [_]i32{ 23, 53, 84 }; // 84 != 83

    const v = try precomputeV(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(!check(v, &x, &r, &z));
}

test "freivalds_negative_weights" {
    const allocator = std.testing.allocator;
    // W = [[-1, 2], [3, -4]]
    const w = [_]i8{ -1, 2, 3, -4 };
    const r = [_]Fp{ Fp{ .val = 5 }, Fp{ .val = 10 } };
    const x = [_]i8{ 3, 7 };
    // z = W * x = [-1*3+2*7, 3*3+(-4)*7] = [11, -19]
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
    const z = [_]i32{ 10, 20, 30 }; // I*x = x

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
    const z = [_]i32{ 23, 53, 84 }; // wrong

    const v = try precomputeV64(allocator, &r, &w, 3, 2);
    defer allocator.free(v);
    try std.testing.expect(!check64(v, &x, &r, &z));
}

test "freivalds_64_negative_weights" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ -1, 2, 3, -4 };
    const r = [_]Fp64{ Fp64{ .val = 5 }, Fp64{ .val = 10 } };
    const x = [_]i8{ 3, 7 };
    const z = [_]i32{ 11, -19 };

    const v = try precomputeV64(allocator, &r, &w, 2, 2);
    defer allocator.free(v);
    try std.testing.expect(check64(v, &x, &r, &z));
}
