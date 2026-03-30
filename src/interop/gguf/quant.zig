const std = @import("std");
const parser = @import("parser.zig");

pub const DequantError = error{
    UnsupportedType,
    InvalidBlockSize,
    UnexpectedEof,
};

/// Dequantize raw GGUF tensor bytes into f32 values.
/// Supports f32 (passthrough), f16, q4_0, q8_0.
/// Returns a newly allocated f32 slice. Caller owns the memory.
pub fn dequantize(
    allocator: std.mem.Allocator,
    raw: []const u8,
    dtype: parser.GGMLType,
    n_elements: u64,
) (DequantError || std.mem.Allocator.Error)![]f32 {
    if (n_elements > std.math.maxInt(usize)) return error.InvalidBlockSize;
    const n: usize = @intCast(n_elements);

    return switch (dtype) {
        .f32 => dequantF32(allocator, raw, n),
        .f16 => dequantF16(allocator, raw, n),
        .q4_0 => dequantQ4_0(allocator, raw, n),
        .q8_0 => dequantQ8_0(allocator, raw, n),
        else => error.UnsupportedType,
    };
}

/// f32 passthrough: read bytes as f32 values and copy.
fn dequantF32(allocator: std.mem.Allocator, raw: []const u8, n: usize) (DequantError || std.mem.Allocator.Error)![]f32 {
    const expected = n * 4;
    if (raw.len < expected) return error.UnexpectedEof;

    const out = try allocator.alloc(f32, n);
    for (0..n) |i| {
        out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    }
    return out;
}

/// f16 → f32: read each half-precision value and convert.
fn dequantF16(allocator: std.mem.Allocator, raw: []const u8, n: usize) (DequantError || std.mem.Allocator.Error)![]f32 {
    const expected = n * 2;
    if (raw.len < expected) return error.UnexpectedEof;

    const out = try allocator.alloc(f32, n);
    for (0..n) |i| {
        const bits = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
        out[i] = @floatCast(@as(f16, @bitCast(bits)));
    }
    return out;
}

/// Q4_0 block layout (block_size = 32 elements, 18 bytes per block):
///   - f16 scale (2 bytes)
///   - 16 bytes of packed 4-bit quantized values (2 values per byte)
///
/// Dequantization: val = (q - 8) * scale
/// where q is the unsigned 4-bit value (0..15) and 8 is the zero point.
fn dequantQ4_0(allocator: std.mem.Allocator, raw: []const u8, n: usize) (DequantError || std.mem.Allocator.Error)![]f32 {
    const QK: usize = 32; // block size in elements
    const BLOCK_BYTES: usize = 18; // 2 (scale) + 16 (quants)

    if (n % QK != 0) return error.InvalidBlockSize;
    const n_blocks = n / QK;
    const expected = n_blocks * BLOCK_BYTES;
    if (raw.len < expected) return error.UnexpectedEof;

    const out = try allocator.alloc(f32, n);

    for (0..n_blocks) |block_idx| {
        const block_start = block_idx * BLOCK_BYTES;
        const block_data = raw[block_start..][0..BLOCK_BYTES];

        // Read f16 scale
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_data[0..2], .little))));

        const quant_data = block_data[2..BLOCK_BYTES];
        const out_offset = block_idx * QK;

        // Each byte contains two 4-bit values: low nibble first, then high nibble
        for (0..16) |j| {
            const byte = quant_data[j];
            const lo: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
            const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;
            out[out_offset + j] = @as(f32, @floatFromInt(lo)) * scale;
            out[out_offset + j + 16] = @as(f32, @floatFromInt(hi)) * scale;
        }
    }

    return out;
}

/// Q8_0 block layout (block_size = 32 elements, 34 bytes per block):
///   - f16 scale (2 bytes)
///   - 32 bytes of signed 8-bit quantized values
///
/// Dequantization: val = q * scale
fn dequantQ8_0(allocator: std.mem.Allocator, raw: []const u8, n: usize) (DequantError || std.mem.Allocator.Error)![]f32 {
    const QK: usize = 32; // block size in elements
    const BLOCK_BYTES: usize = 34; // 2 (scale) + 32 (quants)

    if (n % QK != 0) return error.InvalidBlockSize;
    const n_blocks = n / QK;
    const expected = n_blocks * BLOCK_BYTES;
    if (raw.len < expected) return error.UnexpectedEof;

    const out = try allocator.alloc(f32, n);

    for (0..n_blocks) |block_idx| {
        const block_start = block_idx * BLOCK_BYTES;
        const block_data = raw[block_start..][0..BLOCK_BYTES];

        // Read f16 scale
        const scale: f32 = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, block_data[0..2], .little))));

        const quant_data = block_data[2..BLOCK_BYTES];
        const out_offset = block_idx * QK;

        for (0..QK) |j| {
            const q: i8 = @bitCast(quant_data[j]);
            out[out_offset + j] = @as(f32, @floatFromInt(q)) * scale;
        }
    }

    return out;
}

// ── Tests ───────────────────────────────────────────────────────────

test "quant/f32 passthrough" {
    const values = [_]f32{ 1.0, -2.5, 3.14, 0.0 };
    const raw = std.mem.sliceAsBytes(&values);

    const out = try dequantize(std.testing.allocator, raw, .f32, 4);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), out[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[3], 1e-6);
}

test "quant/f16 conversion" {
    const values = [_]f16{ 1.0, -2.5, 0.0, 0.5 };
    const raw = std.mem.sliceAsBytes(&values);

    const out = try dequantize(std.testing.allocator, raw, .f16, 4);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, 4), out.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -2.5), out[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out[3], 1e-3);
}

test "quant/q8_0 roundtrip" {
    // Create a Q8_0 block manually: 32 elements
    // scale = 0.5 (as f16), quant values = [-4, -3, ..., 27]
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    var block: [BLOCK_BYTES]u8 = undefined;

    // Write scale as f16
    const scale_f16: f16 = 0.5;
    const scale_bytes: [2]u8 = @bitCast(scale_f16);
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // Write quant values: -4, -3, ..., 27
    for (0..QK) |i| {
        const q: i8 = @intCast(@as(i32, @intCast(i)) - 4);
        block[2 + i] = @bitCast(q);
    }

    const out = try dequantize(std.testing.allocator, &block, .q8_0, QK);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, QK), out.len);

    // Check: val = q * 0.5
    for (0..QK) |i| {
        const expected: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 4)) * 0.5;
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-3);
    }
}

test "quant/q4_0 roundtrip" {
    // Create a Q4_0 block manually: 32 elements
    // scale = 1.0 (as f16)
    // 4-bit quants: values 0..15 for low nibbles (first 16 elements)
    //               values 0..15 for high nibbles (second 16 elements)
    // After dequant: val = (q - 8) * scale
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18;
    var block: [BLOCK_BYTES]u8 = undefined;

    // Write scale as f16 = 1.0
    const scale_f16: f16 = 1.0;
    const scale_bytes: [2]u8 = @bitCast(scale_f16);
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // Write packed 4-bit quants
    // For simplicity: all quant values = 8 (zero point), so dequant = 0.0
    for (0..16) |j| {
        block[2 + j] = 0x88; // both nibbles = 8
    }

    const out = try dequantize(std.testing.allocator, &block, .q4_0, QK);
    defer std.testing.allocator.free(out);

    try std.testing.expectEqual(@as(usize, QK), out.len);

    // All values should be 0.0 since (8 - 8) * 1.0 = 0.0
    for (0..QK) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[i], 1e-6);
    }
}

test "quant/q4_0 nonzero values" {
    const QK: usize = 32;
    const BLOCK_BYTES: usize = 18;
    var block: [BLOCK_BYTES]u8 = undefined;

    // scale = 2.0
    const scale_f16: f16 = 2.0;
    const scale_bytes: [2]u8 = @bitCast(scale_f16);
    block[0] = scale_bytes[0];
    block[1] = scale_bytes[1];

    // Byte 0: lo=0, hi=15 → dequant: (0-8)*2=-16, (15-8)*2=14
    block[2] = 0xF0;
    // Fill rest with 8 (zero)
    for (1..16) |j| {
        block[2 + j] = 0x88;
    }

    const out = try dequantize(std.testing.allocator, &block, .q4_0, QK);
    defer std.testing.allocator.free(out);

    // Element 0 (lo nibble of byte 0): (0 - 8) * 2 = -16
    try std.testing.expectApproxEqAbs(@as(f32, -16.0), out[0], 1e-3);
    // Element 16 (hi nibble of byte 0): (15 - 8) * 2 = 14
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), out[16], 1e-3);
    // Element 1 (lo nibble of byte 1): (8 - 8) * 2 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[1], 1e-6);
}

test "quant/unsupported type" {
    const out = dequantize(std.testing.allocator, &[_]u8{}, .q4_1, 0);
    try std.testing.expectError(error.UnsupportedType, out);
}

test "quant/invalid block size" {
    // Q8_0 requires multiples of 32 elements
    var buf: [34]u8 = undefined;
    @memset(&buf, 0);
    const out = dequantize(std.testing.allocator, &buf, .q8_0, 17);
    try std.testing.expectError(error.InvalidBlockSize, out);
}
