//! Prime field F_p with p = 2^32 - 5.
//!
//! All Freivalds checks operate in this field. INT8 values are lifted
//! into F_p, dot products are accumulated in u128 to avoid overflow.
//! Also includes Fp64 (p = 2^61 - 1) for higher security margin.

const std = @import("std");

/// Prime p = 2^32 - 5 = 4,294,967,291.
pub const P: u64 = 4_294_967_291;
const P_u128: u128 = @as(u128, P);

/// Element of the prime field F_p where p = 2^32 - 5.
pub const Fp = struct {
    val: u32,

    pub const ZERO = Fp{ .val = 0 };
    pub const ONE = Fp{ .val = 1 };

    pub fn new(v: u32) Fp {
        return .{ .val = @intCast(@as(u64, v) % P) };
    }

    /// Lift a signed i8 value into F_p.
    /// Maps -128..127 to F_p by taking val mod p (always positive).
    pub fn fromI8(v: i8) Fp {
        const vi: i64 = @intCast(v);
        const reduced: u32 = @intCast(@mod(vi, @as(i64, @intCast(P))));
        return .{ .val = reduced };
    }

    /// Lift a signed i32 value into F_p.
    pub fn fromI32(v: i32) Fp {
        const vi: i64 = @intCast(v);
        const reduced: u32 = @intCast(@mod(vi, @as(i64, @intCast(P))));
        return .{ .val = reduced };
    }

    pub fn add(self: Fp, other: Fp) Fp {
        const sum: u64 = @as(u64, self.val) + @as(u64, other.val);
        return .{ .val = @intCast(sum % P) };
    }

    pub fn sub(self: Fp, other: Fp) Fp {
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

    /// Dot product of two Fp slices. Accumulates in u128.
    pub fn dot(a: []const Fp, b: []const Fp) Fp {
        std.debug.assert(a.len == b.len);
        var acc: u128 = 0;
        for (a, b) |x, y| {
            acc += @as(u128, x.val) * @as(u128, y.val);
        }
        return .{ .val = @intCast(acc % P_u128) };
    }

    /// Dot product of an Fp slice with an i8 slice.
    /// Split positive/negative accumulators to stay in u128.
    pub fn dotFpI8(a: []const Fp, b: []const i8) Fp {
        std.debug.assert(a.len == b.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b) |x, y| {
            const yi: i16 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
        }
        const pos_reduced: u64 = @intCast(pos_acc % P_u128);
        const neg_reduced: u64 = @intCast(neg_acc % P_u128);
        if (pos_reduced >= neg_reduced) {
            return .{ .val = @intCast(pos_reduced - neg_reduced) };
        } else {
            return .{ .val = @intCast(pos_reduced + P - neg_reduced) };
        }
    }

    /// Dot product of an Fp slice with an i32 slice.
    pub fn dotFpI32(a: []const Fp, b: []const i32) Fp {
        std.debug.assert(a.len == b.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b) |x, y| {
            const yi: i64 = @intCast(y);
            if (yi >= 0) {
                pos_acc += @as(u128, x.val) * @as(u128, @intCast(yi));
            } else {
                neg_acc += @as(u128, x.val) * @as(u128, @intCast(-yi));
            }
        }
        const pos_reduced: u64 = @intCast(pos_acc % P_u128);
        const neg_reduced: u64 = @intCast(neg_acc % P_u128);
        if (pos_reduced >= neg_reduced) {
            return .{ .val = @intCast(pos_reduced - neg_reduced) };
        } else {
            return .{ .val = @intCast(pos_reduced + P - neg_reduced) };
        }
    }
};

// ---------------------------------------------------------------------------
// Fp64: prime field with p = 2^61 - 1 (Mersenne prime)
// ---------------------------------------------------------------------------

pub const P64: u64 = (@as(u64, 1) << 61) - 1;

/// Element of the prime field F_p where p = 2^61 - 1 (Mersenne prime).
pub const Fp64 = struct {
    val: u64,

    pub const ZERO = Fp64{ .val = 0 };
    pub const ONE = Fp64{ .val = 1 };

    /// Mersenne reduction for 2^61 - 1.
    pub inline fn reduce(v: u128) u64 {
        const lo: u64 = @intCast(v & @as(u128, P64));
        const hi: u64 = @intCast(v >> 61);
        const sum = lo + hi;
        const lo2 = sum & P64;
        const hi2 = sum >> 61;
        const r = lo2 + hi2;
        return if (r >= P64) r - P64 else r;
    }

    pub fn new(v: u64) Fp64 {
        return .{ .val = reduce(@as(u128, v)) };
    }

    pub fn fromI8(v: i8) Fp64 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            return .{ .val = @intCast(vi) };
        } else {
            return .{ .val = P64 +% @as(u64, @bitCast(vi)) };
        }
    }

    pub fn fromI32(v: i32) Fp64 {
        const vi: i64 = @intCast(v);
        if (vi >= 0) {
            return Fp64.new(@intCast(vi));
        } else {
            return .{ .val = P64 +% @as(u64, @bitCast(vi)) };
        }
    }

    pub fn add(self: Fp64, other: Fp64) Fp64 {
        const sum = self.val + other.val;
        return .{ .val = if (sum >= P64) sum - P64 else sum };
    }

    pub fn sub(self: Fp64, other: Fp64) Fp64 {
        if (self.val >= other.val) {
            return .{ .val = self.val - other.val };
        } else {
            return .{ .val = self.val + P64 - other.val };
        }
    }

    pub fn mul(self: Fp64, other: Fp64) Fp64 {
        const prod: u128 = @as(u128, self.val) * @as(u128, other.val);
        return .{ .val = reduce(prod) };
    }

    pub fn eql(self: Fp64, other: Fp64) bool {
        return self.val == other.val;
    }

    /// Dot product of two Fp64 slices. Batch-reduces every 64 terms.
    pub fn dot(a: []const Fp64, b: []const Fp64) Fp64 {
        std.debug.assert(a.len == b.len);
        var total: u128 = 0;
        var batch: u128 = 0;
        for (a, b, 0..) |x, y, i| {
            batch += @as(u128, x.val) * @as(u128, y.val);
            if ((i & 63) == 63) {
                total = @as(u128, reduce(total)) + @as(u128, reduce(batch));
                batch = 0;
            }
        }
        total = @as(u128, reduce(total)) + @as(u128, reduce(batch));
        return .{ .val = reduce(total) };
    }

    /// Dot product of Fp64 slice with i8 slice.
    pub fn dotFpI8(a: []const Fp64, b: []const i8) Fp64 {
        std.debug.assert(a.len == b.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b, 0..) |x, y, i| {
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
        if (pos >= neg) {
            return .{ .val = pos - neg };
        } else {
            return .{ .val = pos + P64 - neg };
        }
    }

    /// Dot product of Fp64 slice with i32 slice.
    pub fn dotFpI32(a: []const Fp64, b: []const i32) Fp64 {
        std.debug.assert(a.len == b.len);
        var pos_acc: u128 = 0;
        var neg_acc: u128 = 0;
        for (a, b, 0..) |x, y, i| {
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
        if (pos >= neg) {
            return .{ .val = pos - neg };
        } else {
            return .{ .val = pos + P64 - neg };
        }
    }
};

// ===========================================================================
// Tests
// ===========================================================================

test "fp_new_reduces" {
    const a = Fp.new(P + 1);
    try std.testing.expectEqual(@as(u32, 1), a.val);
}

test "fp_from_i8_negative" {
    // -1 mod p = p - 1
    const a = Fp.fromI8(-1);
    try std.testing.expectEqual(@as(u32, @intCast(P - 1)), a.val);
    // -128 mod p = p - 128
    const b = Fp.fromI8(-128);
    try std.testing.expectEqual(@as(u32, @intCast(P - 128)), b.val);
    // 0 mod p = 0
    try std.testing.expectEqual(@as(u32, 0), Fp.fromI8(0).val);
    // 127 mod p = 127
    try std.testing.expectEqual(@as(u32, 127), Fp.fromI8(127).val);
}

test "fp_from_i32_negative" {
    const a = Fp.fromI32(-1);
    try std.testing.expectEqual(@as(u32, @intCast(P - 1)), a.val);
    const b = Fp.fromI32(-19);
    try std.testing.expectEqual(@as(u32, @intCast(P - 19)), b.val);
}

test "fp_add_sub_identity" {
    const a = Fp{ .val = 42 };
    const b = Fp{ .val = 99 };
    // a + b - b == a
    try std.testing.expect(a.add(b).sub(b).eql(a));
    // a - a == 0
    try std.testing.expect(a.sub(a).eql(Fp.ZERO));
}

test "fp_mul_commutativity" {
    const a = Fp{ .val = 123456 };
    const b = Fp{ .val = 789012 };
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
}

test "fp_dot_basic" {
    const a = [_]Fp{ Fp{ .val = 1 }, Fp{ .val = 2 }, Fp{ .val = 3 } };
    const b = [_]Fp{ Fp{ .val = 4 }, Fp{ .val = 5 }, Fp{ .val = 6 } };
    // 1*4 + 2*5 + 3*6 = 32
    const result = Fp.dot(&a, &b);
    try std.testing.expectEqual(@as(u32, 32), result.val);
}

test "fp_dot_fp_i8_basic" {
    const a = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const b = [_]i8{ 7, 8, -1 };
    // 10*7 + 20*8 + 30*(-1) = 70 + 160 - 30 = 200
    const result = Fp.dotFpI8(&a, &b);
    try std.testing.expectEqual(@as(u32, 200), result.val);
}

test "fp_dot_fp_i32_basic" {
    const a = [_]Fp{ Fp{ .val = 10 }, Fp{ .val = 20 }, Fp{ .val = 30 } };
    const b = [_]i32{ 23, 53, 83 };
    // 10*23 + 20*53 + 30*83 = 230 + 1060 + 2490 = 3780
    const result = Fp.dotFpI32(&a, &b);
    try std.testing.expectEqual(@as(u32, 3780), result.val);
}

test "fp64_new_reduces" {
    const a = Fp64.new(P64 + 1);
    try std.testing.expectEqual(@as(u64, 1), a.val);
}

test "fp64_from_i8_negative" {
    const a = Fp64.fromI8(-1);
    try std.testing.expectEqual(P64 - 1, a.val);
}

test "fp64_add_sub_identity" {
    const a = Fp64{ .val = 42 };
    const b = Fp64{ .val = 99 };
    try std.testing.expect(a.add(b).sub(b).eql(a));
}

test "fp64_mul_commutativity" {
    const a = Fp64{ .val = 123456 };
    const b = Fp64{ .val = 789012 };
    try std.testing.expect(a.mul(b).eql(b.mul(a)));
}

test "fp64_dot_fp_i8_basic" {
    const a = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const b = [_]i8{ 7, 8, -1 };
    const result = Fp64.dotFpI8(&a, &b);
    try std.testing.expectEqual(@as(u64, 200), result.val);
}

test "fp64_dot_fp_i32_basic" {
    const a = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
    const b = [_]i32{ 23, 53, 83 };
    const result = Fp64.dotFpI32(&a, &b);
    try std.testing.expectEqual(@as(u64, 3780), result.val);
}
