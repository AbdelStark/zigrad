const std = @import("std");

/// Minimal protobuf wire-format reader for ONNX model parsing.
/// Supports only the wire types needed by the ONNX schema.
pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    fixed32 = 5,
    _,
};

pub const Field = struct {
    number: u32,
    wire_type: WireType,
    data: FieldData,
};

pub const FieldData = union(enum) {
    varint: u64,
    fixed32: u32,
    fixed64: u64,
    bytes: []const u8,
};

pub const DecodeError = error{
    UnexpectedEof,
    InvalidVarint,
    UnsupportedWireType,
    Overflow,
};

/// Read a varint from the byte stream, advancing the position.
pub fn readVarint(data: []const u8, pos: *usize) DecodeError!u64 {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (pos.* < data.len) {
        const byte = data[pos.*];
        pos.* += 1;
        result |= @as(u64, byte & 0x7F) << shift;
        if (byte & 0x80 == 0) return result;
        shift +|= 7;
        if (shift >= 64) return error.InvalidVarint;
    }
    return error.UnexpectedEof;
}

/// Read a signed varint (zigzag decoded).
pub fn readSVarint(data: []const u8, pos: *usize) DecodeError!i64 {
    const v = try readVarint(data, pos);
    // Zigzag decode: (v >>> 1) ^ -(v & 1)
    const half: i64 = @intCast(v >> 1);
    const sign: i64 = -@as(i64, @intCast(v & 1));
    return half ^ sign;
}

/// Read a fixed 32-bit value.
pub fn readFixed32(data: []const u8, pos: *usize) DecodeError!u32 {
    if (pos.* + 4 > data.len) return error.UnexpectedEof;
    const result = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    return result;
}

/// Read a fixed 64-bit value.
pub fn readFixed64(data: []const u8, pos: *usize) DecodeError!u64 {
    if (pos.* + 8 > data.len) return error.UnexpectedEof;
    const result = std.mem.readInt(u64, data[pos.*..][0..8], .little);
    pos.* += 8;
    return result;
}

/// Read a single protobuf field (tag + value).
pub fn readField(data: []const u8, pos: *usize) DecodeError!Field {
    const tag = try readVarint(data, pos);
    const field_number: u32 = @intCast(tag >> 3);
    const wire_type: WireType = @enumFromInt(@as(u3, @truncate(tag)));

    const field_data: FieldData = switch (wire_type) {
        .varint => .{ .varint = try readVarint(data, pos) },
        .fixed32 => .{ .fixed32 = try readFixed32(data, pos) },
        .fixed64 => .{ .fixed64 = try readFixed64(data, pos) },
        .length_delimited => blk: {
            const len: usize = @intCast(try readVarint(data, pos));
            if (pos.* + len > data.len) return error.UnexpectedEof;
            const bytes = data[pos.* .. pos.* + len];
            pos.* += len;
            break :blk .{ .bytes = bytes };
        },
        _ => return error.UnsupportedWireType,
    };

    return .{
        .number = field_number,
        .wire_type = wire_type,
        .data = field_data,
    };
}

/// Iterator over protobuf fields in a byte slice.
pub const FieldIterator = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn next(self: *FieldIterator) ?Field {
        if (self.pos >= self.data.len) return null;
        return readField(self.data, &self.pos) catch null;
    }
};

/// Create an iterator over protobuf fields in a message.
pub fn iterate(data: []const u8) FieldIterator {
    return .{ .data = data };
}

/// Read a packed repeated field of f32 values.
pub fn readPackedF32(data: []const u8, allocator: std.mem.Allocator) ![]f32 {
    const count = data.len / 4;
    const result = try allocator.alloc(f32, count);
    for (0..count) |i| {
        result[i] = @bitCast(std.mem.readInt(u32, data[i * 4 ..][0..4], .little));
    }
    return result;
}

/// Read a packed repeated field of f64 values.
pub fn readPackedF64(data: []const u8, allocator: std.mem.Allocator) ![]f64 {
    const count = data.len / 8;
    const result = try allocator.alloc(f64, count);
    for (0..count) |i| {
        result[i] = @bitCast(std.mem.readInt(u64, data[i * 8 ..][0..8], .little));
    }
    return result;
}

/// Read a packed repeated field of i64 values (varint-encoded).
pub fn readPackedVarint64(data: []const u8, allocator: std.mem.Allocator) ![]i64 {
    var result = std.ArrayListUnmanaged(i64).empty;
    var pos: usize = 0;
    while (pos < data.len) {
        const v = try readVarint(data, &pos);
        try result.append(allocator, @bitCast(v));
    }
    return result.toOwnedSlice(allocator);
}

// ---------- Tests ----------

test "proto/varint roundtrip" {
    // Single byte
    var pos: usize = 0;
    try std.testing.expectEqual(@as(u64, 0), try readVarint(&.{0x00}, &pos));

    pos = 0;
    try std.testing.expectEqual(@as(u64, 1), try readVarint(&.{0x01}, &pos));

    // Multi-byte: 300 = 0xAC 0x02
    pos = 0;
    try std.testing.expectEqual(@as(u64, 300), try readVarint(&.{ 0xAC, 0x02 }, &pos));
}

test "proto/field iterator" {
    // Field 1, varint = 150: tag = (1 << 3) | 0 = 0x08, value = 0x96 0x01
    const data = [_]u8{ 0x08, 0x96, 0x01 };
    var iter = iterate(&data);
    const field = iter.next().?;
    try std.testing.expectEqual(@as(u32, 1), field.number);
    try std.testing.expectEqual(WireType.varint, field.wire_type);
    try std.testing.expectEqual(@as(u64, 150), field.data.varint);
    try std.testing.expect(iter.next() == null);
}

test "proto/length_delimited field" {
    // Field 2, length-delimited = "hi": tag = (2 << 3) | 2 = 0x12, len = 2
    const data = [_]u8{ 0x12, 0x02, 'h', 'i' };
    var iter = iterate(&data);
    const field = iter.next().?;
    try std.testing.expectEqual(@as(u32, 2), field.number);
    try std.testing.expectEqual(WireType.length_delimited, field.wire_type);
    try std.testing.expectEqualStrings("hi", field.data.bytes);
}
