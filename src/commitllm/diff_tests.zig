//! Differential tests: load JSON vectors from the Rust generator and
//! verify bit-exact compatibility with the Zig implementation.

const std = @import("std");
const field = @import("field.zig");
const freivalds = @import("freivalds.zig");
const merkle = @import("merkle.zig");
const silu_mod = @import("silu.zig");
const requantize_mod = @import("requantize.zig");
const toy_model = @import("toy_model.zig");

const Fp = field.Fp;

const fixtures_dir = "tests/fixtures/commitllm";

fn readFixture(allocator: std.mem.Allocator, filename: []const u8) ![]const u8 {
    const file = std.fs.cwd().openFile(filename, .{}) catch |e| {
        std.log.warn("Skipping differential test: cannot open {s}: {s}", .{ filename, @errorName(e) });
        return error.SkipTest;
    };
    defer file.close();
    return try file.readToEndAlloc(allocator, 50 * 1024 * 1024);
}

fn hexToHash(hex: []const u8) ![32]u8 {
    if (hex.len != 64) return error.InvalidHexLength;
    var result: [32]u8 = undefined;
    for (0..32) |i| {
        result[i] = std.fmt.parseInt(u8, hex[i * 2 .. i * 2 + 2], 16) catch return error.InvalidHex;
    }
    return result;
}

// ===========================================================================
// Field differential tests
// ===========================================================================

test "diff_field_from_i8" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/field_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("from_i8").?.array;
    for (cases.items) |item| {
        const input: i8 = @intCast(item.object.get("input").?.integer);
        const expected: u32 = @intCast(item.object.get("expected").?.integer);
        const actual = Fp.fromI8(input);
        try std.testing.expectEqual(expected, actual.val);
    }
}

test "diff_field_from_i32" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/field_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("from_i32").?.array;
    for (cases.items) |item| {
        const input: i32 = @intCast(item.object.get("input").?.integer);
        const expected: u32 = @intCast(item.object.get("expected").?.integer);
        const actual = Fp.fromI32(input);
        try std.testing.expectEqual(expected, actual.val);
    }
}

test "diff_field_add_sub_mul" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/field_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("add_sub_mul").?.array;
    for (cases.items) |item| {
        const a_val: u32 = @intCast(item.object.get("a").?.integer);
        const b_val: u32 = @intCast(item.object.get("b").?.integer);
        const exp_add: u32 = @intCast(item.object.get("add").?.integer);
        const exp_sub: u32 = @intCast(item.object.get("sub").?.integer);
        const exp_mul: u32 = @intCast(item.object.get("mul").?.integer);

        const a = Fp.new(a_val);
        const b = Fp.new(b_val);
        try std.testing.expectEqual(exp_add, a.add(b).val);
        try std.testing.expectEqual(exp_sub, a.sub(b).val);
        try std.testing.expectEqual(exp_mul, a.mul(b).val);
    }
}

test "diff_field_dot_fp_i8" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/field_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("dot_fp_i8").?.array;
    for (cases.items) |item| {
        const a_arr = item.object.get("a").?.array;
        const b_arr = item.object.get("b").?.array;
        const expected: u32 = @intCast(item.object.get("expected").?.integer);

        const a_fp = try allocator.alloc(Fp, a_arr.items.len);
        defer allocator.free(a_fp);
        for (a_arr.items, 0..) |v, i| {
            a_fp[i] = Fp.new(@intCast(v.integer));
        }

        const b_i8 = try allocator.alloc(i8, b_arr.items.len);
        defer allocator.free(b_i8);
        for (b_arr.items, 0..) |v, i| {
            b_i8[i] = @intCast(v.integer);
        }

        const actual = Fp.dotFpI8(a_fp, b_i8);
        try std.testing.expectEqual(expected, actual.val);
    }
}

test "diff_field_dot_fp_i32" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/field_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("dot_fp_i32").?.array;
    for (cases.items) |item| {
        const a_arr = item.object.get("a").?.array;
        const b_arr = item.object.get("b").?.array;
        const expected: u32 = @intCast(item.object.get("expected").?.integer);

        const a_fp = try allocator.alloc(Fp, a_arr.items.len);
        defer allocator.free(a_fp);
        for (a_arr.items, 0..) |v, i| {
            a_fp[i] = Fp.new(@intCast(v.integer));
        }

        const b_i32 = try allocator.alloc(i32, b_arr.items.len);
        defer allocator.free(b_i32);
        for (b_arr.items, 0..) |v, i| {
            b_i32[i] = @intCast(v.integer);
        }

        const actual = Fp.dotFpI32(a_fp, b_i32);
        try std.testing.expectEqual(expected, actual.val);
    }
}

// ===========================================================================
// Freivalds differential tests
// ===========================================================================

test "diff_freivalds" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/freivalds_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("cases").?.array;
    for (cases.items) |item| {
        const rows: usize = @intCast(item.object.get("rows").?.integer);
        const cols: usize = @intCast(item.object.get("cols").?.integer);
        const check_passes = item.object.get("check_passes").?.bool;

        // Parse r
        const r_arr = item.object.get("r").?.array;
        const r = try allocator.alloc(Fp, r_arr.items.len);
        defer allocator.free(r);
        for (r_arr.items, 0..) |v, i| {
            r[i] = Fp{ .val = @intCast(v.integer) };
        }

        // Parse weight
        const w_arr = item.object.get("weight").?.array;
        const weight = try allocator.alloc(i8, w_arr.items.len);
        defer allocator.free(weight);
        for (w_arr.items, 0..) |v, i| {
            weight[i] = @intCast(v.integer);
        }

        // Parse x
        const x_arr = item.object.get("x").?.array;
        const x = try allocator.alloc(i8, x_arr.items.len);
        defer allocator.free(x);
        for (x_arr.items, 0..) |v, i| {
            x[i] = @intCast(v.integer);
        }

        // Parse z
        const z_arr = item.object.get("z").?.array;
        const z = try allocator.alloc(i32, z_arr.items.len);
        defer allocator.free(z);
        for (z_arr.items, 0..) |v, i| {
            z[i] = @intCast(v.integer);
        }

        // Parse expected v
        const v_arr = item.object.get("v").?.array;
        const expected_v = try allocator.alloc(u32, v_arr.items.len);
        defer allocator.free(expected_v);
        for (v_arr.items, 0..) |v, i| {
            expected_v[i] = @intCast(v.integer);
        }

        // Compute v and compare
        const computed_v = try freivalds.precomputeV(allocator, r, weight, rows, cols);
        defer allocator.free(computed_v);

        for (computed_v, expected_v) |cv, ev| {
            try std.testing.expectEqual(ev, cv.val);
        }

        // Check Freivalds
        const result = freivalds.check(computed_v, x, r, z);
        try std.testing.expectEqual(check_passes, result);
    }
}

// ===========================================================================
// Merkle differential tests
// ===========================================================================

test "diff_merkle" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/merkle_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const cases = root.object.get("cases").?.array;
    for (cases.items) |item| {
        const leaves_arr = item.object.get("leaves").?.array;
        const expected_root_hex = item.object.get("root").?.string;

        const expected_root = try hexToHash(expected_root_hex);

        // Parse leaves
        const leaves = try allocator.alloc(merkle.Hash, leaves_arr.items.len);
        defer allocator.free(leaves);
        for (leaves_arr.items, 0..) |v, i| {
            leaves[i] = try hexToHash(v.string);
        }

        // Compute root and compare
        const computed_root = try merkle.computeRoot(allocator, leaves);
        try std.testing.expectEqualSlices(u8, &expected_root, &computed_root);
    }
}

// ===========================================================================
// SiLU differential tests
// ===========================================================================

test "diff_silu_lut" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/silu_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const lut_arr = root.object.get("lut_unit").?.array;
    const zig_lut = silu_mod.buildSiluLut(1.0);

    for (lut_arr.items, 0..) |v, i| {
        const rust_val: f32 = @floatCast(v.float);
        try std.testing.expectApproxEqAbs(rust_val, zig_lut[i], 1e-6);
    }
}

test "diff_silu_h" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/silu_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const h_cases = root.object.get("h_cases").?.array;
    for (h_cases.items) |item| {
        const g_arr = item.object.get("g_acc").?.array;
        const u_arr = item.object.get("u_acc").?.array;
        const h_arr = item.object.get("h").?.array;

        const g = try allocator.alloc(i32, g_arr.items.len);
        defer allocator.free(g);
        for (g_arr.items, 0..) |v, i| {
            g[i] = @intCast(v.integer);
        }

        const u = try allocator.alloc(i32, u_arr.items.len);
        defer allocator.free(u);
        for (u_arr.items, 0..) |v, i| {
            u[i] = @intCast(v.integer);
        }

        const zig_h = try silu_mod.computeHUnitScale(allocator, g, u);
        defer allocator.free(zig_h);

        for (zig_h, h_arr.items) |zh, v| {
            const rust_h: i8 = @intCast(v.integer);
            try std.testing.expectEqual(rust_h, zh);
        }
    }
}

// ===========================================================================
// E2E differential tests
// ===========================================================================

test "diff_e2e_matmul" {
    const allocator = std.testing.allocator;
    const content = readFixture(allocator, fixtures_dir ++ "/e2e_vectors.json") catch return;
    defer allocator.free(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value;

    const config = root.object.get("config").?.object;
    const hidden_dim: usize = @intCast(config.get("hidden_dim").?.integer);

    // Parse Wq
    const wq_arr = root.object.get("model_layer0_wq").?.array;
    const wq = try allocator.alloc(i8, wq_arr.items.len);
    defer allocator.free(wq);
    for (wq_arr.items, 0..) |v, i| {
        wq[i] = @intCast(v.integer);
    }

    // Parse input
    const input_arr = root.object.get("input").?.array;
    const input = try allocator.alloc(i8, input_arr.items.len);
    defer allocator.free(input);
    for (input_arr.items, 0..) |v, i| {
        input[i] = @intCast(v.integer);
    }

    // Parse expected Q
    const q_arr = root.object.get("layer0_q").?.array;
    const expected_q = try allocator.alloc(i32, q_arr.items.len);
    defer allocator.free(expected_q);
    for (q_arr.items, 0..) |v, i| {
        expected_q[i] = @intCast(v.integer);
    }

    // Compute Q = Wq * input
    const computed_q = try toy_model.matmulI32(allocator, wq, input, hidden_dim, hidden_dim);
    defer allocator.free(computed_q);

    // Compare
    for (computed_q, expected_q) |cq, eq| {
        try std.testing.expectEqual(eq, cq);
    }
}
