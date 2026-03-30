const std = @import("std");

/// GGUF container format parser.
///
/// Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
///
/// The GGUF format is a binary container for tensor data with metadata.
/// All multi-byte values are little-endian. The layout is:
///
///   Header:      magic (4B) | version (u32) | tensor_count (u64) | metadata_kv_count (u64)
///   Metadata:    repeated { key (string) | value_type (u32) | value }
///   Tensor info: repeated { name (string) | n_dims (u32) | dims (u64 * n_dims) | type (u32) | offset (u64) }
///   Padding:     to alignment boundary (default 32 bytes)
///   Tensor data: raw tensor bytes at declared offsets from data section start

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;
pub const MAX_DIMS: usize = 8;

pub const ParseError = error{
    InvalidMagic,
    UnsupportedVersion,
    UnexpectedEof,
    InvalidString,
    InvalidMetadataType,
    InvalidTensorType,
    InvalidDimCount,
    Overflow,
    OutOfMemory,
};

// ── Metadata value types ────────────────────────────────────────────

pub const MetadataValueType = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    bool_ = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
    _,
};

pub const MetadataValue = union(MetadataValueType) {
    uint8: u8,
    int8: i8,
    uint16: u16,
    int16: i16,
    uint32: u32,
    int32: i32,
    float32: f32,
    bool_: bool,
    string: []const u8,
    array: MetadataArray,
    uint64: u64,
    int64: i64,
    float64: f64,
};

pub const MetadataArray = struct {
    elem_type: MetadataValueType,
    values: []const MetadataValue,
};

pub const MetadataKV = struct {
    key: []const u8,
    value: MetadataValue,
};

// ── Tensor types ────────────────────────────────────────────────────

pub const GGMLType = enum(u32) {
    f32 = 0,
    f16 = 1,
    q4_0 = 2,
    q4_1 = 3,
    // q4_2 = 4, (removed)
    // q4_3 = 5, (removed)
    q5_0 = 6,
    q5_1 = 7,
    q8_0 = 8,
    q8_1 = 9,
    q2_k = 10,
    q3_k = 11,
    q4_k = 12,
    q5_k = 13,
    q6_k = 14,
    q8_k = 15,
    iq2_xxs = 16,
    iq2_xs = 17,
    iq3_xxs = 18,
    iq1_s = 19,
    iq4_nl = 20,
    iq3_s = 21,
    iq2_s = 22,
    iq4_xs = 23,
    i8 = 24,
    i16 = 25,
    i32 = 26,
    i64 = 27,
    f64 = 28,
    iq1_m = 29,
    bf16 = 30,
    _,
};

/// Returns the block size in elements for a given GGML type.
pub fn blockSize(t: GGMLType) u64 {
    return switch (t) {
        .f32, .f16, .bf16, .f64 => 1,
        .i8, .i16, .i32, .i64 => 1,
        .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .q8_1 => 32,
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .q8_k => 256,
        else => 1,
    };
}

/// Returns the byte size of a single block for a given GGML type.
pub fn typeSize(t: GGMLType) u64 {
    return switch (t) {
        .f32 => 4,
        .f16 => 2,
        .bf16 => 2,
        .f64 => 8,
        .i8 => 1,
        .i16 => 2,
        .i32 => 4,
        .i64 => 8,
        .q4_0 => 2 + 16, // f16 scale + 16 bytes of 4-bit quants (32 values)
        .q4_1 => 2 + 2 + 16, // f16 scale + f16 min + 16 bytes
        .q5_0 => 2 + 4 + 16, // f16 scale + 4 bytes high bits + 16 bytes
        .q5_1 => 2 + 2 + 4 + 16, // f16 scale + f16 min + 4 bytes + 16 bytes
        .q8_0 => 2 + 32, // f16 scale + 32 bytes of 8-bit quants
        .q8_1 => 4 + 4 + 32, // f32 scale + f32 sum + 32 bytes
        else => 0, // unsupported types return 0
    };
}

// ── Tensor info ─────────────────────────────────────────────────────

pub const TensorInfo = struct {
    name: []const u8,
    n_dims: u32,
    dimensions: [MAX_DIMS]u64,
    dtype: GGMLType,
    offset: u64, // offset from start of tensor data section

    pub fn shape(self: *const TensorInfo) []const u64 {
        return self.dimensions[0..self.n_dims];
    }

    pub fn elemCount(self: *const TensorInfo) u64 {
        var count: u64 = 1;
        for (self.dimensions[0..self.n_dims]) |d| {
            count *|= d;
        }
        return count;
    }

    pub fn byteSize(self: *const TensorInfo) u64 {
        const n_elem = self.elemCount();
        const bs = blockSize(self.dtype);
        const ts = typeSize(self.dtype);
        if (bs == 0 or ts == 0) return 0;
        return (n_elem / bs) * ts;
    }
};

// ── Parsed GGUF file ────────────────────────────────────────────────

pub const GGUFFile = struct {
    version: u32,
    alignment: u64,
    metadata: []MetadataKV,
    tensors: []TensorInfo,
    data_offset: u64, // byte offset where tensor data begins in the file
    data: []const u8, // the full file data (for tensor loading)

    pub fn deinit(self: *GGUFFile, allocator: std.mem.Allocator) void {
        for (self.metadata) |*kv| {
            freeMetadataValue(allocator, &kv.value);
        }
        allocator.free(self.metadata);
        allocator.free(self.tensors);
    }

    /// Look up a metadata value by key. Returns null if not found.
    pub fn getMetadata(self: *const GGUFFile, key: []const u8) ?*const MetadataValue {
        for (self.metadata) |*kv| {
            if (std.mem.eql(u8, kv.key, key)) return &kv.value;
        }
        return null;
    }

    /// Get the alignment, checking the metadata for an override.
    pub fn getAlignment(self: *const GGUFFile) u64 {
        return self.alignment;
    }

    /// Get the raw bytes for a tensor's data region.
    pub fn tensorData(self: *const GGUFFile, info: *const TensorInfo) ParseError![]const u8 {
        const start = self.data_offset + info.offset;
        const size = info.byteSize();
        if (start + size > self.data.len) return error.UnexpectedEof;
        return self.data[start..start + size];
    }
};

fn freeMetadataValue(allocator: std.mem.Allocator, value: *const MetadataValue) void {
    switch (value.*) {
        .array => |arr| {
            for (arr.values) |*v| {
                freeMetadataValue(allocator, v);
            }
            allocator.free(arr.values);
        },
        else => {},
    }
}

// ── Low-level readers ───────────────────────────────────────────────

const Reader = struct {
    data: []const u8,
    pos: usize,

    fn init(data: []const u8) Reader {
        return .{ .data = data, .pos = 0 };
    }

    fn remaining(self: *const Reader) usize {
        if (self.pos >= self.data.len) return 0;
        return self.data.len - self.pos;
    }

    fn readBytes(self: *Reader, n: usize) ParseError![]const u8 {
        if (self.remaining() < n) return error.UnexpectedEof;
        const slice = self.data[self.pos..][0..n];
        self.pos += n;
        return slice;
    }

    fn readU8(self: *Reader) ParseError!u8 {
        const bytes = try self.readBytes(1);
        return bytes[0];
    }

    fn readI8(self: *Reader) ParseError!i8 {
        return @bitCast(try self.readU8());
    }

    fn readU16(self: *Reader) ParseError!u16 {
        const bytes = try self.readBytes(2);
        return std.mem.readInt(u16, bytes[0..2], .little);
    }

    fn readI16(self: *Reader) ParseError!i16 {
        return @bitCast(try self.readU16());
    }

    fn readU32(self: *Reader) ParseError!u32 {
        const bytes = try self.readBytes(4);
        return std.mem.readInt(u32, bytes[0..4], .little);
    }

    fn readI32(self: *Reader) ParseError!i32 {
        return @bitCast(try self.readU32());
    }

    fn readU64(self: *Reader) ParseError!u64 {
        const bytes = try self.readBytes(8);
        return std.mem.readInt(u64, bytes[0..8], .little);
    }

    fn readI64(self: *Reader) ParseError!i64 {
        return @bitCast(try self.readU64());
    }

    fn readF32(self: *Reader) ParseError!f32 {
        const bytes = try self.readBytes(4);
        return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
    }

    fn readF64(self: *Reader) ParseError!f64 {
        const bytes = try self.readBytes(8);
        return @bitCast(std.mem.readInt(u64, bytes[0..8], .little));
    }

    fn readBool(self: *Reader) ParseError!bool {
        return (try self.readU8()) != 0;
    }

    /// Read a GGUF string: u64 length + raw bytes (no null terminator).
    fn readString(self: *Reader) ParseError![]const u8 {
        const len = try self.readU64();
        if (len > std.math.maxInt(usize)) return error.Overflow;
        const n: usize = @intCast(len);
        return try self.readBytes(n);
    }

    fn alignTo(self: *Reader, alignment: u64) void {
        if (alignment == 0) return;
        const a: usize = @intCast(alignment);
        const remainder = self.pos % a;
        if (remainder != 0) {
            const pad = a - remainder;
            if (self.pos + pad <= self.data.len) {
                self.pos += pad;
            }
        }
    }
};

// ── Metadata parsing ────────────────────────────────────────────────

fn readMetadataValue(reader: *Reader, allocator: std.mem.Allocator, vtype: MetadataValueType) ParseError!MetadataValue {
    return switch (vtype) {
        .uint8 => .{ .uint8 = try reader.readU8() },
        .int8 => .{ .int8 = try reader.readI8() },
        .uint16 => .{ .uint16 = try reader.readU16() },
        .int16 => .{ .int16 = try reader.readI16() },
        .uint32 => .{ .uint32 = try reader.readU32() },
        .int32 => .{ .int32 = try reader.readI32() },
        .float32 => .{ .float32 = try reader.readF32() },
        .bool_ => .{ .bool_ = try reader.readBool() },
        .string => .{ .string = try reader.readString() },
        .uint64 => .{ .uint64 = try reader.readU64() },
        .int64 => .{ .int64 = try reader.readI64() },
        .float64 => .{ .float64 = try reader.readF64() },
        .array => {
            const elem_type_raw = try reader.readU32();
            const elem_type: MetadataValueType = @enumFromInt(elem_type_raw);
            const count = try reader.readU64();
            if (count > std.math.maxInt(usize)) return error.Overflow;
            const n: usize = @intCast(count);

            const values = try allocator.alloc(MetadataValue, n);
            errdefer allocator.free(values);

            for (0..n) |i| {
                values[i] = try readMetadataValue(reader, allocator, elem_type);
            }

            return .{ .array = .{
                .elem_type = elem_type,
                .values = values,
            } };
        },
        _ => return error.InvalidMetadataType,
    };
}

fn readMetadataKV(reader: *Reader, allocator: std.mem.Allocator) ParseError!MetadataKV {
    const key = try reader.readString();
    const vtype_raw = try reader.readU32();
    const vtype: MetadataValueType = @enumFromInt(vtype_raw);
    const value = try readMetadataValue(reader, allocator, vtype);
    return .{ .key = key, .value = value };
}

// ── Tensor info parsing ─────────────────────────────────────────────

fn readTensorInfo(reader: *Reader) ParseError!TensorInfo {
    const name = try reader.readString();
    const n_dims = try reader.readU32();
    if (n_dims > MAX_DIMS) return error.InvalidDimCount;

    var dims: [MAX_DIMS]u64 = .{0} ** MAX_DIMS;
    for (0..n_dims) |i| {
        dims[i] = try reader.readU64();
    }

    const dtype_raw = try reader.readU32();
    const dtype: GGMLType = @enumFromInt(dtype_raw);
    const offset = try reader.readU64();

    return .{
        .name = name,
        .n_dims = n_dims,
        .dimensions = dims,
        .dtype = dtype,
        .offset = offset,
    };
}

// ── Top-level parser ────────────────────────────────────────────────

/// Parse a GGUF file from a byte buffer. The returned `GGUFFile` references
/// slices into `data` for string and tensor data — the caller must keep
/// `data` alive for the lifetime of the result.
pub fn parse(data: []const u8, allocator: std.mem.Allocator) ParseError!GGUFFile {
    var reader = Reader.init(data);

    // Header
    const magic = try reader.readU32();
    if (magic != GGUF_MAGIC) return error.InvalidMagic;

    const version = try reader.readU32();
    if (version != GGUF_VERSION_2 and version != GGUF_VERSION_3) return error.UnsupportedVersion;

    const tensor_count = try reader.readU64();
    const metadata_kv_count = try reader.readU64();

    if (tensor_count > std.math.maxInt(usize)) return error.Overflow;
    if (metadata_kv_count > std.math.maxInt(usize)) return error.Overflow;

    const n_tensors: usize = @intCast(tensor_count);
    const n_metadata: usize = @intCast(metadata_kv_count);

    // Metadata
    const metadata = try allocator.alloc(MetadataKV, n_metadata);
    errdefer allocator.free(metadata);

    var alignment: u64 = GGUF_DEFAULT_ALIGNMENT;

    for (0..n_metadata) |i| {
        metadata[i] = try readMetadataKV(&reader, allocator);
        // Check for alignment override
        if (std.mem.eql(u8, metadata[i].key, "general.alignment")) {
            switch (metadata[i].value) {
                .uint32 => |v| alignment = v,
                .uint64 => |v| alignment = v,
                else => {},
            }
        }
    }

    // Tensor info table
    const tensors = try allocator.alloc(TensorInfo, n_tensors);
    errdefer allocator.free(tensors);

    for (0..n_tensors) |i| {
        tensors[i] = try readTensorInfo(&reader);
    }

    // Align to start of tensor data
    reader.alignTo(alignment);
    const data_offset = reader.pos;

    return .{
        .version = version,
        .alignment = alignment,
        .metadata = metadata,
        .tensors = tensors,
        .data_offset = data_offset,
        .data = data,
    };
}

// ── Tests ───────────────────────────────────────────────────────────

test "parser/reader primitives" {
    var buf: [32]u8 = undefined;
    // Write a u32 = 0x12345678
    std.mem.writeInt(u32, buf[0..4], 0x12345678, .little);
    // Write a u64 = 0xAABBCCDDEEFF0011
    std.mem.writeInt(u64, buf[4..12], 0xAABBCCDDEEFF0011, .little);

    var reader = Reader.init(&buf);
    const v32 = try reader.readU32();
    try std.testing.expectEqual(@as(u32, 0x12345678), v32);
    const v64 = try reader.readU64();
    try std.testing.expectEqual(@as(u64, 0xAABBCCDDEEFF0011), v64);
}

test "parser/reader string" {
    // GGUF string: u64 length + bytes
    var buf: [32]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], 5, .little); // length = 5
    @memcpy(buf[8..13], "hello");

    var reader = Reader.init(buf[0..13]);
    const s = try reader.readString();
    try std.testing.expectEqualStrings("hello", s);
}

test "parser/minimal gguf file" {
    // Construct a minimal valid GGUF file in memory:
    // 1 metadata KV (string), 1 tensor (f32, 2x3)
    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);

    const writer = buf.writer(std.testing.allocator);

    // Magic
    try writer.writeInt(u32, GGUF_MAGIC, .little);
    // Version
    try writer.writeInt(u32, GGUF_VERSION_3, .little);
    // Tensor count
    try writer.writeInt(u64, 1, .little);
    // Metadata KV count
    try writer.writeInt(u64, 1, .little);

    // Metadata KV: key="general.name", type=string, value="test-model"
    try writeString(writer, "general.name");
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.string), .little);
    try writeString(writer, "test-model");

    // Tensor info: name="weight.0", ndims=2, dims=[2,3], type=f32, offset=0
    try writeString(writer, "weight.0");
    try writer.writeInt(u32, 2, .little); // ndims
    try writer.writeInt(u64, 2, .little); // dim 0
    try writer.writeInt(u64, 3, .little); // dim 1
    try writer.writeInt(u32, @intFromEnum(GGMLType.f32), .little);
    try writer.writeInt(u64, 0, .little); // offset

    // Align to 32 bytes
    const current = buf.items.len;
    const aligned = ((current + 31) / 32) * 32;
    for (0..aligned - current) |_| {
        try writer.writeByte(0);
    }

    // Tensor data: 6 f32 values
    const tensor_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try writer.writeAll(std.mem.sliceAsBytes(&tensor_data));

    // Parse
    var gguf = try parse(buf.items, std.testing.allocator);
    defer gguf.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u32, GGUF_VERSION_3), gguf.version);
    try std.testing.expectEqual(@as(usize, 1), gguf.metadata.len);
    try std.testing.expectEqual(@as(usize, 1), gguf.tensors.len);

    // Check metadata
    try std.testing.expectEqualStrings("general.name", gguf.metadata[0].key);
    try std.testing.expectEqualStrings("test-model", gguf.metadata[0].value.string);

    // Check tensor info
    const ti = &gguf.tensors[0];
    try std.testing.expectEqualStrings("weight.0", ti.name);
    try std.testing.expectEqual(@as(u32, 2), ti.n_dims);
    try std.testing.expectEqual(@as(u64, 2), ti.dimensions[0]);
    try std.testing.expectEqual(@as(u64, 3), ti.dimensions[1]);
    try std.testing.expectEqual(GGMLType.f32, ti.dtype);
    try std.testing.expectEqual(@as(u64, 6), ti.elemCount());
    try std.testing.expectEqual(@as(u64, 24), ti.byteSize());

    // Check tensor data access
    const tdata = try gguf.tensorData(ti);
    try std.testing.expectEqual(@as(usize, 24), tdata.len);
    // Read first f32 value
    const first: f32 = @bitCast(std.mem.readInt(u32, tdata[0..4], .little));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), first, 1e-6);
    // Read last f32 value
    const last: f32 = @bitCast(std.mem.readInt(u32, tdata[20..24], .little));
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), last, 1e-6);
}

test "parser/metadata types" {
    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header: 0 tensors, 4 metadata KVs
    try writer.writeInt(u32, GGUF_MAGIC, .little);
    try writer.writeInt(u32, GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 4, .little);

    // KV 1: uint32
    try writeString(writer, "arch.block_count");
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.uint32), .little);
    try writer.writeInt(u32, 32, .little);

    // KV 2: float32
    try writeString(writer, "arch.rope_freq");
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.float32), .little);
    try writer.writeAll(&std.mem.toBytes(@as(f32, 10000.0)));

    // KV 3: bool
    try writeString(writer, "arch.use_parallel");
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.bool_), .little);
    try writer.writeByte(1);

    // KV 4: array of uint32
    try writeString(writer, "tokenizer.scores");
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.array), .little);
    try writer.writeInt(u32, @intFromEnum(MetadataValueType.uint32), .little);
    try writer.writeInt(u64, 3, .little); // 3 elements
    try writer.writeInt(u32, 10, .little);
    try writer.writeInt(u32, 20, .little);
    try writer.writeInt(u32, 30, .little);

    var gguf = try parse(buf.items, std.testing.allocator);
    defer gguf.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 4), gguf.metadata.len);

    // uint32
    try std.testing.expectEqual(@as(u32, 32), gguf.metadata[0].value.uint32);

    // float32
    try std.testing.expectApproxEqAbs(@as(f32, 10000.0), gguf.metadata[1].value.float32, 1e-2);

    // bool
    try std.testing.expect(gguf.metadata[2].value.bool_);

    // array
    const arr = gguf.metadata[3].value.array;
    try std.testing.expectEqual(@as(usize, 3), arr.values.len);
    try std.testing.expectEqual(@as(u32, 10), arr.values[0].uint32);
    try std.testing.expectEqual(@as(u32, 20), arr.values[1].uint32);
    try std.testing.expectEqual(@as(u32, 30), arr.values[2].uint32);

    // getMetadata lookup
    const v = gguf.getMetadata("arch.block_count");
    try std.testing.expect(v != null);
    try std.testing.expectEqual(@as(u32, 32), v.?.uint32);

    // Missing key
    try std.testing.expect(gguf.getMetadata("nonexistent") == null);
}

test "parser/invalid magic" {
    var buf: [24]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 0xDEADBEEF, .little);
    std.mem.writeInt(u32, buf[4..8], GGUF_VERSION_3, .little);
    std.mem.writeInt(u64, buf[8..16], 0, .little);
    std.mem.writeInt(u64, buf[16..24], 0, .little);

    const result = parse(&buf, std.testing.allocator);
    try std.testing.expectError(error.InvalidMagic, result);
}

test "parser/unsupported version" {
    var buf: [24]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], GGUF_MAGIC, .little);
    std.mem.writeInt(u32, buf[4..8], 99, .little);
    std.mem.writeInt(u64, buf[8..16], 0, .little);
    std.mem.writeInt(u64, buf[16..24], 0, .little);

    const result = parse(&buf, std.testing.allocator);
    try std.testing.expectError(error.UnsupportedVersion, result);
}

test "parser/multiple tensors" {
    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header: 3 tensors, 0 metadata
    try writer.writeInt(u32, GGUF_MAGIC, .little);
    try writer.writeInt(u32, GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 3, .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor 0: f32 [4], offset 0
    try writeString(writer, "layer.0.weight");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 4, .little);
    try writer.writeInt(u32, @intFromEnum(GGMLType.f32), .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor 1: f16 [2, 4], offset 16
    try writeString(writer, "layer.0.bias");
    try writer.writeInt(u32, 2, .little);
    try writer.writeInt(u64, 2, .little);
    try writer.writeInt(u64, 4, .little);
    try writer.writeInt(u32, @intFromEnum(GGMLType.f16), .little);
    try writer.writeInt(u64, 16, .little);

    // Tensor 2: q8_0 [32], offset 32
    try writeString(writer, "layer.1.weight");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 32, .little);
    try writer.writeInt(u32, @intFromEnum(GGMLType.q8_0), .little);
    try writer.writeInt(u64, 32, .little);

    // Pad to alignment (no tensor data needed for this parse test)
    const current = buf.items.len;
    const aligned = ((current + 31) / 32) * 32;
    for (0..aligned - current) |_| {
        try writer.writeByte(0);
    }

    var gguf = try parse(buf.items, std.testing.allocator);
    defer gguf.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 3), gguf.tensors.len);

    try std.testing.expectEqualStrings("layer.0.weight", gguf.tensors[0].name);
    try std.testing.expectEqual(GGMLType.f32, gguf.tensors[0].dtype);
    try std.testing.expectEqual(@as(u64, 4), gguf.tensors[0].elemCount());

    try std.testing.expectEqualStrings("layer.0.bias", gguf.tensors[1].name);
    try std.testing.expectEqual(GGMLType.f16, gguf.tensors[1].dtype);
    try std.testing.expectEqual(@as(u64, 8), gguf.tensors[1].elemCount());

    try std.testing.expectEqualStrings("layer.1.weight", gguf.tensors[2].name);
    try std.testing.expectEqual(GGMLType.q8_0, gguf.tensors[2].dtype);
    try std.testing.expectEqual(@as(u64, 32), gguf.tensors[2].elemCount());
    try std.testing.expectEqual(@as(u64, 34), gguf.tensors[2].byteSize()); // 2 + 32
}

// ── Test helpers ────────────────────────────────────────────────────

fn writeString(writer: anytype, s: []const u8) !void {
    try writer.writeInt(u64, s.len, .little);
    try writer.writeAll(s);
}
