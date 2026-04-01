//! Prime fields for Freivalds algebraic verification.
//!
//! Three fields of increasing security margin:
//!   - Fp:    p = 2^32 - 5         (false-accept ≈ 2.3e-10)
//!   - Fp64:  p = 2^61 - 1         (Mersenne, false-accept ≈ 4.3e-19)
//!   - Fp128: p = 2^127 - 1        (Mersenne, false-accept ≈ 5.9e-39)
//!
//! INT8 values are lifted into the field; dot products are accumulated
//! in wide integers (u128 / U256) with split positive/negative tracking
//! to avoid overflow before reduction.

const std = @import("std");

// ═══════════════════════════════════════════════════════════════════════
// Fp: prime field with p = 2^32 - 5
// ═══════════════════════════════════════════════════════════════════════

/// Prime p = 2^32 - 5 = 4,294,967,291.
pub const P: u64 = 4_294_967_291;
const P_u128: u128 = @as(u128, P);

/// Element of F_p where p = 2^32 - 5.
///
/// All transformer shell matmul checks use this field by default.
/// The false-accept probability per check is 1/p ≈ 2.3e-10.
pub const Fp = struct {
    val: u32,

    pub const ZERO = Fp{ .val = 0 };
    pub const ONE = Fp{ .val = 1 };

    /// Reduce a u32 into F_p.
    pub fn new(v: u32) Fp {
        return .{ .val = @intCast(@as(u64, v) % P) };
    }

    /// Lift a signed i8 into F_p via Euclidean remainder.
    pub fn fromI8(v: i8) Fp {
        const vi: i64 = @intCast(v);
        return .{ .val = @intCast(@mod(vi, @as(i64, @intCast(P)))) };
    }

    /// Lift a signed i32 into F_p via Euclidean remainder.
    pub fn fromI32(v: i32) Fp {
        const vi: i64 = @intCast(v);
        return .{ .val = @intCast(@mod(vi, @as(i64, @intCast(P)))) };
    }

    pub fn add(self: Fp, other: Fp) Fp {
        const sum: u64 = @as(u64, self.val) + @as(u64, other.val);
        return .{ .val = @intCast(sum % P) };
    }

    pub fn sub(self: Fp, other: Fp) Fp {
        // Add P before subtracting to prevent underflow.
        const diff: u64 = @as(u64, self.val) + P - @as(u64, other.val);
        return .{ .val = @intCast(diff % P) };
    }

    pub fn mul(self: Fp, other: Fp) Fp {
        const prod: u64 = @as(u64, self.val) * @as(u64, other.val);
        return .{ .val = @intCast(prod % P) };
    }

    pub fn eql(self: Fp, other: Fp) bool {
        return self.val == other.val;
    }

    /// Inner product ⟨a, b⟩ in F_p. Accumulates in u128.
    ///
    /// Overflow budget: each term ≤ (P-1)² < 2^64.
    /// u128 holds 2^128, so safe for n < 2^64 terms — never reachable.
    pub fn dot(a: []const Fp, b: []const Fp) Fp {
        std.debug.assert(a.len == b.len);
        var acc: u128 = 0;
        for (a, b) |x, y| {
            acc += @as(u128, x.val) * @as(u128, y.val);
        }
        return .{ .val = @intCast(acc % P_u128) };
    }

    /// Mixed dot product ⟨a_Fp, b_i8⟩. Splits positive/negative to avoid
    /// signed arithmetic in the accumulator.
    ///
    /// Overflow budget: each |term| ≤ (P-1) × 128 < 2^39.
    /// u128 safe for n < 2^89 terms — never reachable.
    pub fn dotFpI8(a: []const Fp, b_slice: []const i8) Fp {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b_slice) |x, y| {
            const yi: i16 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
        }
        const pos_reduced: u64 = @intCast(pos_acc % P_u128);
        const neg_reduced: u64 = @intCast(neg_acc % P_u128);
        return if (pos_reduced >= neg_reduced)
            .{ .val = @intCast(pos_reduced - neg_reduced) }
        else
            .{ .val = @intCast(pos_reduced + P - neg_reduced) };
    }

    /// Mixed dot product ⟨a_Fp, b_i32⟩.
    ///
    /// Overflow budget: each |term| ≤ (P-1) × 2^31 < 2^63.
    /// u128 safe for n < 2^65 terms — never reachable.
    pub fn dotFpI32(a: []const Fp, b_slice: []const i32) Fp {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b_slice) |x, y| {
            const yi: i64 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
        }
        const pos_reduced: u64 = @intCast(pos_acc % P_u128);
        const neg_reduced: u64 = @intCast(neg_acc % P_u128);
        return if (pos_reduced >= neg_reduced)
            .{ .val = @intCast(pos_reduced - neg_reduced) }
        else
            .{ .val = @intCast(pos_reduced + P - neg_reduced) };
    }

    pub fn format(self: Fp, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Fp({})", .{self.val});
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Fp64: prime field with p = 2^61 - 1 (Mersenne prime)
// ═══════════════════════════════════════════════════════════════════════

/// Mersenne prime p = 2^61 - 1.
pub const P64: u64 = (@as(u64, 1) << 61) - 1;

/// Element of F_p where p = 2^61 - 1.
///
/// Uses the Mersenne structure for fast modular reduction:
///   x mod (2^61-1) = (x >> 61) + (x & (2^61-1)), with at most one
///   additional subtraction.
pub const Fp64 = struct {
    val: u64,

    pub const ZERO = Fp64{ .val = 0 };
    pub const ONE = Fp64{ .val = 1 };

    /// Mersenne reduction: x mod (2^61-1).
    pub inline fn reduce(v: u128) u64 {
        const lo: u64 = @intCast(v & @as(u128, P64));
        const hi: u64 = @intCast(v >> 61);
        const sum = lo + hi;
        // One more round: sum might still be >= P64.
        const lo2 = sum & P64;
        const hi2 = sum >> 61;
        const r = lo2 + hi2;
        return if (r >= P64) r - P64 else r;
    }

    pub fn new(v: u64) Fp64 {
        return .{ .val = reduce(@as(u128, v)) };
    }

    /// Lift i8 into F_p64. Uses subtraction from P64 for negative values.
    pub fn fromI8(v: i8) Fp64 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            return .{ .val = @intCast(vi) };
        } else {
            // -vi fits in u64 and is < P64, so P64 - (-vi) is in range.
            return .{ .val = P64 - @as(u64, @intCast(-vi)) };
        }
    }

    /// Lift i32 into F_p64. Uses subtraction from P64 for negative values.
    pub fn fromI32(v: i32) Fp64 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            // Values up to 2^31-1 are already < P64, no reduction needed.
            return .{ .val = @intCast(vi) };
        } else {
            return .{ .val = P64 - @as(u64, @intCast(-vi)) };
        }
    }

    pub fn add(self: Fp64, other: Fp64) Fp64 {
        std.debug.assert(self.val < P64);
        std.debug.assert(other.val < P64);
        const sum = self.val + other.val;
        return .{ .val = if (sum >= P64) sum - P64 else sum };
    }

    pub fn sub(self: Fp64, other: Fp64) Fp64 {
        std.debug.assert(self.val < P64);
        std.debug.assert(other.val < P64);
        return if (self.val >= other.val)
            .{ .val = self.val - other.val }
        else
            .{ .val = self.val + P64 - other.val };
    }

    pub fn mul(self: Fp64, other: Fp64) Fp64 {
        const prod: u128 = @as(u128, self.val) * @as(u128, other.val);
        return .{ .val = reduce(prod) };
    }

    pub fn eql(self: Fp64, other: Fp64) bool {
        return self.val == other.val;
    }

    /// Inner product ⟨a, b⟩ in F_p64. Batch-reduces every 64 terms.
    ///
    /// Each term ≤ (P64-1)² ≈ 2^122. After 64 terms: 64×2^122 = 2^128.
    /// Must reduce before hitting u128 overflow.
    pub fn dot(a: []const Fp64, b_slice: []const Fp64) Fp64 {
        std.debug.assert(a.len == b_slice.len);
        var total: u128 = 0;
        var batch: u128 = 0;
        for (a, b_slice, 0..) |x, y, i| {
            batch += @as(u128, x.val) * @as(u128, y.val);
            if ((i & 63) == 63) {
                total = @as(u128, reduce(total)) + @as(u128, reduce(batch));
                batch = 0;
            }
        }
        total = @as(u128, reduce(total)) + @as(u128, reduce(batch));
        return .{ .val = reduce(total) };
    }

    /// Mixed dot product ⟨a_Fp64, b_i8⟩. Reduces every 2^20 terms.
    ///
    /// Each |term| ≤ (P64-1) × 128 < 2^68.
    /// After 2^20 terms: 2^68 × 2^20 = 2^88 — safe in u128.
    pub fn dotFpI8(a: []const Fp64, b_slice: []const i8) Fp64 {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b_slice, 0..) |x, y, i| {
            const yi: i16 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
            if ((i & 0xFFFFF) == 0xFFFFF) {
                pos_acc = @as(u128, reduce(pos_acc));
                neg_acc = @as(u128, reduce(neg_acc));
            }
        }
        const pos = reduce(pos_acc);
        const neg = reduce(neg_acc);
        return if (pos >= neg)
            .{ .val = pos - neg }
        else
            .{ .val = pos + P64 - neg };
    }

    /// Mixed dot product ⟨a_Fp64, b_i32⟩. Reduces every 2^20 terms.
    ///
    /// Each |term| ≤ (P64-1) × 2^31 < 2^92.
    /// After 2^20 terms: 2^92 × 2^20 = 2^112 — safe in u128.
    pub fn dotFpI32(a: []const Fp64, b_slice: []const i32) Fp64 {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b_slice, 0..) |x, y, i| {
            const yi: i64 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
            if ((i & 0xFFFFF) == 0xFFFFF) {
                pos_acc = @as(u128, reduce(pos_acc));
                neg_acc = @as(u128, reduce(neg_acc));
            }
        }
        const pos = reduce(pos_acc);
        const neg = reduce(neg_acc);
        return if (pos >= neg)
            .{ .val = pos - neg }
        else
            .{ .val = pos + P64 - neg };
    }

    pub fn format(self: Fp64, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.print("Fp64({})", .{self.val});
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Fp128: prime field with p = 2^127 - 1 (Mersenne prime)
// ═══════════════════════════════════════════════════════════════════════

/// Mersenne prime p = 2^127 - 1.
pub const P128: u128 = (@as(u128, 1) << 127) - 1;

/// 256-bit unsigned integer as (hi, lo) pair of u128.
/// Represents the value hi × 2^128 + lo.
pub const U256 = struct {
    hi: u128,
    lo: u128,

    pub const ZERO = U256{ .hi = 0, .lo = 0 };

    /// Multiply two u128 values, producing a 256-bit result.
    pub inline fn mul128(a: u128, b: u128) U256 {
        const a_lo: u128 = @as(u64, @truncate(a));
        const a_hi: u128 = a >> 64;
        const b_lo: u128 = @as(u64, @truncate(b));
        const b_hi: u128 = b >> 64;

        const ll = a_lo * b_lo;
        const lh = a_lo * b_hi;
        const hl = a_hi * b_lo;
        const hh = a_hi * b_hi;

        // Combine: result = hh × 2^128 + (lh + hl) × 2^64 + ll
        const mid = lh +% hl;
        const mid_carry: u128 = if (mid < lh) @as(u128, 1) << 64 else 0;
        const lo = ll +% (mid << 64);
        const lo_carry: u128 = if (lo < ll) 1 else 0;
        const hi = hh + (mid >> 64) + mid_carry + lo_carry;

        return .{ .hi = hi, .lo = lo };
    }

    /// Add another U256 to this one (in-place).
    ///
    /// SAFETY: caller must ensure the running sum does not overflow the hi
    /// lane (u128). In practice this means reducing the accumulator before
    /// hi exceeds ~2^126 — see the batch intervals in Fp128.dot/dotFpI8/dotFpI32.
    pub fn addAssign(self: *U256, other: U256) void {
        const new_lo = self.lo +% other.lo;
        const carry: u128 = if (new_lo < self.lo) 1 else 0;
        self.lo = new_lo;
        // Detect hi overflow in debug builds. In release, this is a no-op.
        if (std.debug.runtime_safety) {
            const prev_hi = self.hi;
            self.hi += other.hi + carry;
            std.debug.assert(self.hi >= prev_hi); // overflow check
        } else {
            self.hi += other.hi + carry;
        }
    }

    /// Reduce mod 2^127 - 1.
    pub fn reduceMersenne127(self: U256) u128 {
        const mask = P128;
        const lower = self.lo & mask;
        const upper_from_lo = self.lo >> 127;
        // hi × 2^128 = hi × 2 × 2^127, contributing hi × 2 to upper.
        const upper = self.hi * 2 + upper_from_lo;

        // Fold upper: it could still be large.
        const upper_lo = upper & mask;
        const upper_hi = upper >> 127;
        const folded_upper = upper_lo + upper_hi;

        var r = lower + folded_upper;
        if (r >= (@as(u128, 1) << 127)) {
            r = (r & mask) + (r >> 127);
        }
        if (r >= mask) {
            r -= mask;
        }
        return r;
    }
};

/// Element of F_p where p = 2^127 - 1.
///
/// The highest security margin. Uses U256 for wide multiplication.
pub const Fp128 = struct {
    val: u128,

    pub const ZERO = Fp128{ .val = 0 };
    pub const ONE = Fp128{ .val = 1 };

    inline fn reduce(v: u128) u128 {
        const lo = v & P128;
        const hi = v >> 127;
        const r = lo + hi;
        return if (r >= P128) r - P128 else r;
    }

    pub fn new(v: u128) Fp128 {
        return .{ .val = reduce(v) };
    }

    pub fn fromI8(v: i8) Fp128 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            return .{ .val = @intCast(vi) };
        } else {
            return .{ .val = P128 - @as(u128, @intCast(-vi)) };
        }
    }

    pub fn fromI32(v: i32) Fp128 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            return .{ .val = @intCast(vi) };
        } else {
            return .{ .val = P128 - @as(u128, @intCast(-vi)) };
        }
    }

    pub fn add(self: Fp128, other: Fp128) Fp128 {
        std.debug.assert(self.val < P128);
        std.debug.assert(other.val < P128);
        const sum = self.val + other.val;
        return .{ .val = if (sum >= P128) sum - P128 else sum };
    }

    pub fn sub(self: Fp128, other: Fp128) Fp128 {
        std.debug.assert(self.val < P128);
        std.debug.assert(other.val < P128);
        return if (self.val >= other.val)
            .{ .val = self.val - other.val }
        else
            .{ .val = self.val + P128 - other.val };
    }

    pub fn mul(self: Fp128, other: Fp128) Fp128 {
        const prod = U256.mul128(self.val, other.val);
        return .{ .val = prod.reduceMersenne127() };
    }

    pub fn eql(self: Fp128, other: Fp128) bool {
        return self.val == other.val;
    }

    /// Inner product ⟨a, b⟩ in Fp128.
    ///
    /// Each product U256.mul128(x, y) has hi ≤ 2^126. Accumulating k products
    /// before reduction requires hi to stay below 2^128. Safe batch size: 2.
    /// We reduce every 2 terms for correctness at all vector lengths.
    pub fn dot(a: []const Fp128, b_slice: []const Fp128) Fp128 {
        std.debug.assert(a.len == b_slice.len);
        var acc = U256.ZERO;
        for (a, b_slice, 0..) |x, y, i| {
            const prod = U256.mul128(x.val, y.val);
            acc.addAssign(prod);
            if ((i & 1) == 1) {
                const reduced = acc.reduceMersenne127();
                acc = .{ .hi = 0, .lo = reduced };
            }
        }
        return .{ .val = acc.reduceMersenne127() };
    }

    /// Mixed dot product ⟨a_Fp128, b_i8⟩. Split pos/neg in U256.
    ///
    /// Each term: a.val < 2^127, |b| ≤ 128, product hi < 2^(127+7-128) = 2^6.
    /// Safe to accumulate up to 2^(128-6) = 2^122 terms. We reduce every
    /// 2^16 terms for a comfortable margin at all realistic model sizes.
    pub fn dotFpI8(a: []const Fp128, b_slice: []const i8) Fp128 {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc = U256.ZERO;
        var neg_acc = U256.ZERO;
        for (a, b_slice, 0..) |x, y, i| {
            const yi: i16 = @intCast(y);
            if (yi >= 0) {
                pos_acc.addAssign(U256.mul128(x.val, @intCast(yi)));
            } else {
                neg_acc.addAssign(U256.mul128(x.val, @intCast(-yi)));
            }
            if ((i & 0xFFFF) == 0xFFFF) {
                pos_acc = .{ .hi = 0, .lo = pos_acc.reduceMersenne127() };
                neg_acc = .{ .hi = 0, .lo = neg_acc.reduceMersenne127() };
            }
        }
        const pos = pos_acc.reduceMersenne127();
        const neg = neg_acc.reduceMersenne127();
        return if (pos >= neg)
            .{ .val = pos - neg }
        else
            .{ .val = pos + P128 - neg };
    }

    /// Mixed dot product ⟨a_Fp128, b_i32⟩.
    ///
    /// Each term: a.val < 2^127, |b| < 2^31, product hi < 2^(127+31-128) = 2^30.
    /// Safe to accumulate 2^(128-30) = 2^98 terms. Reduce every 2^16 terms.
    pub fn dotFpI32(a: []const Fp128, b_slice: []const i32) Fp128 {
        std.debug.assert(a.len == b_slice.len);
        var pos_acc = U256.ZERO;
        var neg_acc = U256.ZERO;
        for (a, b_slice, 0..) |x, y, i| {
            const yi: i64 = @intCast(y);
            if (yi >= 0) {
                pos_acc.addAssign(U256.mul128(x.val, @intCast(yi)));
            } else {
                neg_acc.addAssign(U256.mul128(x.val, @intCast(-yi)));
            }
            if ((i & 0xFFFF) == 0xFFFF) {
                pos_acc = .{ .hi = 0, .lo = pos_acc.reduceMersenne127() };
                neg_acc = .{ .hi = 0, .lo = neg_acc.reduceMersenne127() };
            }
        }
        const pos = pos_acc.reduceMersenne127();
        const neg = neg_acc.reduceMersenne127();
        return if (pos >= neg)
            .{ .val = pos - neg }
        else
            .{ .val = pos + P128 - neg };
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "fp_new_reduces" {
    try std.testing.expectEqual(@as(u32, 1), Fp.new(@as(u32, @intCast(P + 1))).val);
    try std.testing.expectEqual(@as(u32, 0), Fp.new(0).val);
    try std.testing.expectEqual(@as(u32, @intCast(P - 1)), Fp.new(@as(u32, @intCast(P - 1))).val);
}

test "fp_from_i8_boundary" {
    try std.testing.expectEqual(@as(u32, @intCast(P - 1)), Fp.fromI8(-1).val);
    try std.testing.expectEqual(@as(u32, @intCast(P - 128)), Fp.fromI8(-128).val);
    try std.testing.expectEqual(@as(u32, 0), Fp.fromI8(0).val);
    try std.testing.expectEqual(@as(u32, 127), Fp.fromI8(127).val);
    try std.testing.expectEqual(@as(u32, @intCast(P - 64)), Fp.fromI8(-64).val);
}

test "fp_from_i32_boundary" {
    try std.testing.expectEqual(@as(u32, @intCast(P - 1)), Fp.fromI32(-1).val);
    try std.testing.expectEqual(@as(u32, @intCast(P - 19)), Fp.fromI32(-19).val);
    // i32 min = -2_147_483_648
    const min_expected = Fp.fromI32(std.math.minInt(i32));
    _ = min_expected; // just verify it doesn't panic
    // i32 max
    const max_expected = Fp.fromI32(std.math.maxInt(i32));
    _ = max_expected;
}

test "fp_arithmetic_laws" {
    const a = Fp{ .val = 42 };
    const b = Fp{ .val = 99 };
    const c = Fp{ .val = 7 };
    // Additive identity
    try std.testing.expect(a.add(Fp.ZERO).eql(a));
    // Additive inverse
    try std.testing.expect(a.add(a.sub(a)).eql(a));
    // a + b - b == a
    try std.testing.expect(a.add(b).sub(b).eql(a));
    // Commutativity
    try std.testing.expect(a.add(b).eql(b.add(a)));
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
    // Multiplicative identity
    try std.testing.expect(a.mul(Fp.ONE).eql(a));
    // Distributive: a*(b+c) == a*b + a*c
    try std.testing.expect(a.mul(b.add(c)).eql(a.mul(b).add(a.mul(c))));
    // (-1) * (-1) == 1
    const neg_one = Fp.fromI8(-1);
    try std.testing.expect(neg_one.mul(neg_one).eql(Fp.ONE));
}

test "fp_dot_basic" {
    const a = [_]Fp{ Fp{ .val = 1 }, Fp{ .val = 2 }, Fp{ .val = 3 } };
    const b = [_]Fp{ Fp{ .val = 4 }, Fp{ .val = 5 }, Fp{ .val = 6 } };
    try std.testing.expectEqual(@as(u32, 32), Fp.dot(&a, &b).val);
}

test "fp_dot_fp_i8_matches_dot" {
    // Verify dotFpI8 gives the same result as manually lifting and dotting.
    const a = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const b_i8 = [_]i8{ 7, -3, 5 };
    const b_fp = [_]Fp{ Fp.fromI8(7), Fp.fromI8(-3), Fp.fromI8(5) };
    const via_mixed = Fp.dotFpI8(&a, &b_i8);
    const via_lift = Fp.dot(&a, &b_fp);
    try std.testing.expect(via_mixed.eql(via_lift));
}

test "fp_dot_fp_i32_basic" {
    const a = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const b = [_]i32{ 23, 53, 83 };
    try std.testing.expectEqual(@as(u32, 3780), Fp.dotFpI32(&a, &b).val);
}

// --- Fp64 tests ---

test "fp64_from_i8_all_boundary" {
    try std.testing.expectEqual(P64 - 1, Fp64.fromI8(-1).val);
    try std.testing.expectEqual(P64 - 128, Fp64.fromI8(-128).val);
    try std.testing.expectEqual(@as(u64, 0), Fp64.fromI8(0).val);
    try std.testing.expectEqual(@as(u64, 127), Fp64.fromI8(127).val);
    try std.testing.expectEqual(P64 - 64, Fp64.fromI8(-64).val);
}

test "fp64_from_i32_boundary" {
    try std.testing.expectEqual(P64 - 1, Fp64.fromI32(-1).val);
    // -2^31 should map to P64 - 2^31
    const min_val = std.math.minInt(i32); // -2_147_483_648
    try std.testing.expectEqual(P64 - 2_147_483_648, Fp64.fromI32(min_val).val);
    try std.testing.expectEqual(@as(u64, 2_147_483_647), Fp64.fromI32(std.math.maxInt(i32)).val);
}

test "fp64_arithmetic_laws" {
    const a = Fp64{ .val = 42 };
    const b = Fp64{ .val = 99 };
    try std.testing.expect(a.add(b).sub(b).eql(a));
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
    try std.testing.expect(a.mul(Fp64.ONE).eql(a));
    const neg_one = Fp64.fromI8(-1);
    try std.testing.expect(neg_one.mul(neg_one).eql(Fp64.ONE));
}

test "fp64_dot_fp_i8" {
    const a = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const b = [_]i8{ 7, 8, -1 };
    try std.testing.expectEqual(@as(u64, 200), Fp64.dotFpI8(&a, &b).val);
}

test "fp64_dot_fp_i32" {
    const a = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const b = [_]i32{ 23, 53, 83 };
    try std.testing.expectEqual(@as(u64, 3780), Fp64.dotFpI32(&a, &b).val);
}

// --- Fp128 tests ---

test "fp128_from_i8_boundary" {
    try std.testing.expectEqual(P128 - 1, Fp128.fromI8(-1).val);
    try std.testing.expectEqual(P128 - 128, Fp128.fromI8(-128).val);
    try std.testing.expectEqual(@as(u128, 0), Fp128.fromI8(0).val);
    try std.testing.expectEqual(@as(u128, 127), Fp128.fromI8(127).val);
}

test "fp128_arithmetic_laws" {
    const a = Fp128{ .val = 42 };
    const b = Fp128{ .val = 99 };
    try std.testing.expect(a.add(b).sub(b).eql(a));
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
    try std.testing.expect(a.mul(Fp128.ONE).eql(a));
    const neg_one = Fp128.fromI8(-1);
    try std.testing.expect(neg_one.mul(neg_one).eql(Fp128.ONE));
}

test "fp128_dot_correctness" {
    const a = [_]Fp128{ Fp128{ .val = 10 }, Fp128{ .val = 20 }, Fp128{ .val = 30 } };
    const b = [_]Fp128{ Fp128{ .val = 4 }, Fp128{ .val = 5 }, Fp128{ .val = 6 } };
    // 10*4 + 20*5 + 30*6 = 40 + 100 + 180 = 320
    try std.testing.expectEqual(@as(u128, 320), Fp128.dot(&a, &b).val);
}

test "fp128_dot_fp_i8" {
    const a = [_]Fp128{ Fp128{ .val = 10 }, Fp128{ .val = 20 }, Fp128{ .val = 30 } };
    const b = [_]i8{ 7, 8, -1 };
    try std.testing.expectEqual(@as(u128, 200), Fp128.dotFpI8(&a, &b).val);
}

test "fp128_dot_fp_i32" {
    const a = [_]Fp128{ Fp128{ .val = 10 }, Fp128{ .val = 20 }, Fp128{ .val = 30 } };
    const b = [_]i32{ 23, 53, 83 };
    try std.testing.expectEqual(@as(u128, 3780), Fp128.dotFpI32(&a, &b).val);
}

test "fp_dot_empty" {
    const empty_fp: []const Fp = &.{};
    try std.testing.expect(Fp.dot(empty_fp, empty_fp).eql(Fp.ZERO));
    const empty_fp64: []const Fp64 = &.{};
    try std.testing.expect(Fp64.dot(empty_fp64, empty_fp64).eql(Fp64.ZERO));
    const empty_fp128: []const Fp128 = &.{};
    try std.testing.expect(Fp128.dot(empty_fp128, empty_fp128).eql(Fp128.ZERO));
}

test "u256_mul_basic" {
    // 2^64 * 2^64 = 2^128 → hi=1, lo=0
    const big: u128 = @as(u128, 1) << 64;
    const result = U256.mul128(big, big);
    try std.testing.expectEqual(@as(u128, 1), result.hi);
    try std.testing.expectEqual(@as(u128, 0), result.lo);

    // Small multiplication
    const r2 = U256.mul128(3, 7);
    try std.testing.expectEqual(@as(u128, 0), r2.hi);
    try std.testing.expectEqual(@as(u128, 21), r2.lo);
}

test "u256_reduce_mersenne127" {
    // P128 should reduce to 0
    const v = U256{ .hi = 0, .lo = P128 };
    try std.testing.expectEqual(@as(u128, 0), v.reduceMersenne127());

    // P128 + 1 should reduce to 1
    const v2 = U256{ .hi = 0, .lo = P128 + 1 };
    try std.testing.expectEqual(@as(u128, 1), v2.reduceMersenne127());
}
