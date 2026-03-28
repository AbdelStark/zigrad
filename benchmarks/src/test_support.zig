const std = @import("std");

pub fn countElements(shape: []const usize) usize {
    var total: usize = 1;
    for (shape) |dim| total *= dim;
    return total;
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

pub fn makeDeterministicSlice(
    allocator: std.mem.Allocator,
    count: usize,
    seed: u64,
) ![]f32 {
    const values = try allocator.alloc(f32, count);
    for (values, 0..) |*value, index| {
        const mixed = splitmix64(seed +% @as(u64, index));
        const normalized = (@as(f64, @floatFromInt(mixed % 10_000)) / 10_000.0) - 0.5;
        value.* = @as(f32, @floatCast(normalized * 0.25));
    }
    return values;
}

pub fn makeGraphEdgeIndex(
    allocator: std.mem.Allocator,
    node_count: usize,
    fanout: usize,
) ![]usize {
    const edge_count = node_count * fanout;
    const values = try allocator.alloc(usize, edge_count * 2);

    for (0..node_count) |node| {
        for (0..fanout) |slot| {
            const edge_index = (node * fanout) + slot;
            values[edge_index] = node;
            values[edge_count + edge_index] = switch (slot) {
                0 => node,
                1 => if (node == 0) node_count - 1 else node - 1,
                2 => (node + 1) % node_count,
                else => (node + 2) % node_count,
            };
        }
    }

    return values;
}

pub fn expectApproxEqAbsRelSlices(
    comptime T: type,
    expected: []const T,
    actual: []const T,
    abs_tolerance: T,
    rel_tolerance: T,
) !void {
    if (@typeInfo(T) != .float) {
        @compileError("expectApproxEqAbsRelSlices only works with float types");
    }

    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual, 0..) |expected_value, actual_value, index| {
        if (!approxEqAbsRel(T, expected_value, actual_value, abs_tolerance, rel_tolerance)) {
            std.debug.print(
                "slices differ at index {d}: expected={e}, actual={e}, abs_tolerance={e}, rel_tolerance={e}\n",
                .{ index, expected_value, actual_value, abs_tolerance, rel_tolerance },
            );
            return error.TestExpectedApproxEqAbsRel;
        }
    }
}

fn approxEqAbsRel(
    comptime T: type,
    expected: T,
    actual: T,
    abs_tolerance: T,
    rel_tolerance: T,
) bool {
    const diff = @abs(expected - actual);
    if (diff <= abs_tolerance) return true;
    const scale = @max(@abs(expected), @abs(actual));
    return diff <= rel_tolerance * scale;
}

test "expectApproxEqAbsRelSlices accepts small drift" {
    const expected = [_]f32{ 0.0, 1.0, -2.0 };
    const actual = [_]f32{ 1e-6, 1.0 + (2e-6), -2.0 - (2e-6) };

    try expectApproxEqAbsRelSlices(f32, &expected, &actual, 1e-5, 1e-5);
    try std.testing.expect(!approxEqAbsRel(f32, expected[1], actual[1], 1e-8, 1e-8));
}
