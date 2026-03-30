const std = @import("std");
const parser = @import("parser.zig");
const quant = @import("quant.zig");
const zg = @import("../../zigrad.zig");

const NDArray = zg.NDArray;
const DeviceReference = zg.DeviceReference;

pub const LoadError = parser.ParseError || quant.DequantError || error{
    TensorDataTruncated,
    UnsupportedDType,
    ShapeTooLarge,
};

/// A loaded tensor: name → f32 NDArray, with the original GGUF dtype recorded.
pub const LoadedTensor = struct {
    array: NDArray(f32),
    original_dtype: parser.GGMLType,
};

/// Result of loading tensors from a GGUF file.
/// Caller must call `deinit()` to free all resources.
pub const TensorMap = struct {
    entries: std.StringArrayHashMapUnmanaged(LoadedTensor),
    allocator: std.mem.Allocator,
    /// The parsed GGUF file (holds metadata, tensor info, references into data).
    gguf: parser.GGUFFile,

    pub fn deinit(self: *TensorMap, device: DeviceReference) void {
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.array.deinit(device);
        }
        self.entries.deinit(self.allocator);
        self.gguf.deinit(self.allocator);
    }

    pub fn count(self: *const TensorMap) usize {
        return self.entries.count();
    }

    pub fn get(self: *const TensorMap, name: []const u8) ?*const LoadedTensor {
        return self.entries.getPtr(name);
    }

    pub fn getMetadata(self: *const TensorMap, key: []const u8) ?*const parser.MetadataValue {
        return self.gguf.getMetadata(key);
    }
};

/// Options for tensor loading.
pub const LoadOptions = struct {
    /// If non-null, only load tensors whose names are in this set.
    /// Useful for partial loading / debugging.
    filter: ?[]const []const u8 = null,
};

/// Load all tensors from a GGUF file into device memory as f32 NDArrays.
///
/// The `data` buffer must remain valid for the lifetime of the returned
/// `TensorMap` (metadata strings reference into it). The tensor data
/// itself is copied to device memory.
///
/// Supported GGUF types: f32, f16, q4_0, q8_0.
/// Unsupported types are skipped with a diagnostic log.
pub fn loadTensors(
    allocator: std.mem.Allocator,
    data: []const u8,
    device: DeviceReference,
    options: LoadOptions,
) LoadError!TensorMap {
    var gguf = try parser.parse(data, allocator);
    errdefer gguf.deinit(allocator);

    var entries = std.StringArrayHashMapUnmanaged(LoadedTensor){};
    errdefer {
        var iter = entries.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.array.deinit(device);
        }
        entries.deinit(allocator);
    }

    for (gguf.tensors) |*ti| {
        // Apply filter if specified
        if (options.filter) |filter| {
            var found = false;
            for (filter) |name| {
                if (std.mem.eql(u8, ti.name, name)) {
                    found = true;
                    break;
                }
            }
            if (!found) continue;
        }

        // Check if dtype is supported
        if (!isSupportedDtype(ti.dtype)) {
            std.log.warn("GGUF: skipping tensor '{s}' with unsupported type {d}", .{
                ti.name, @intFromEnum(ti.dtype),
            });
            continue;
        }

        // Get raw tensor bytes
        const raw = gguf.tensorData(ti) catch |err| switch (err) {
            error.UnexpectedEof => return error.TensorDataTruncated,
            else => return err,
        };

        // Dequantize to f32
        const f32_data = try quant.dequantize(allocator, raw, ti.dtype, ti.elemCount());
        defer allocator.free(f32_data);

        // Build shape as []const usize
        const n_dims = ti.n_dims;
        if (n_dims > parser.MAX_DIMS) return error.ShapeTooLarge;

        var shape_buf: [parser.MAX_DIMS]usize = undefined;
        for (0..n_dims) |d| {
            shape_buf[d] = @intCast(ti.dimensions[d]);
        }
        const shape = shape_buf[0..n_dims];

        // Create NDArray on device
        const array = NDArray(f32).from_slice(f32_data, shape, device) catch {
            return error.ShapeTooLarge;
        };

        try entries.put(allocator, ti.name, .{
            .array = array,
            .original_dtype = ti.dtype,
        });
    }

    return .{
        .entries = entries,
        .allocator = allocator,
        .gguf = gguf,
    };
}

fn isSupportedDtype(dtype: parser.GGMLType) bool {
    return switch (dtype) {
        .f32, .f16, .q4_0, .q8_0 => true,
        else => false,
    };
}

// ── Tests ───────────────────────────────────────────────────────────

test "loader/load f32 tensors" {
    const device_mod = @import("../../device.zig");
    var host = device_mod.HostDevice.init();
    const device = host.reference();

    // Build a minimal GGUF file with one f32 tensor [2, 3]
    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header
    try writer.writeInt(u32, parser.GGUF_MAGIC, .little);
    try writer.writeInt(u32, parser.GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 1, .little); // 1 tensor
    try writer.writeInt(u64, 0, .little); // 0 metadata

    // Tensor info
    try writeString(writer, "weights");
    try writer.writeInt(u32, 2, .little); // ndims
    try writer.writeInt(u64, 2, .little); // dim 0
    try writer.writeInt(u64, 3, .little); // dim 1
    try writer.writeInt(u32, @intFromEnum(parser.GGMLType.f32), .little);
    try writer.writeInt(u64, 0, .little); // offset

    // Align
    padToAlignment(&buf, std.testing.allocator, 32);

    // Tensor data: 6 f32 values
    const tensor_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try writer.writeAll(std.mem.sliceAsBytes(&tensor_values));

    var result = try loadTensors(std.testing.allocator, buf.items, device, .{});
    defer result.deinit(device);

    try std.testing.expectEqual(@as(usize, 1), result.count());

    const t = result.get("weights");
    try std.testing.expect(t != null);
    try std.testing.expectEqual(parser.GGMLType.f32, t.?.original_dtype);

    const data = t.?.array.get_data();
    try std.testing.expectEqual(@as(usize, 6), data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), data[5], 1e-6);

    // Shape check
    const shape = t.?.array.shape.slice();
    try std.testing.expectEqual(@as(usize, 2), shape.len);
    try std.testing.expectEqual(@as(u64, 2), shape[0]);
    try std.testing.expectEqual(@as(u64, 3), shape[1]);
}

test "loader/load f16 tensors" {
    const device_mod = @import("../../device.zig");
    var host = device_mod.HostDevice.init();
    const device = host.reference();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header
    try writer.writeInt(u32, parser.GGUF_MAGIC, .little);
    try writer.writeInt(u32, parser.GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 1, .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor info: f16 [4]
    try writeString(writer, "bias");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 4, .little);
    try writer.writeInt(u32, @intFromEnum(parser.GGMLType.f16), .little);
    try writer.writeInt(u64, 0, .little);

    padToAlignment(&buf, std.testing.allocator, 32);

    // Tensor data: 4 f16 values
    const f16_values = [_]f16{ 1.0, -0.5, 0.0, 2.0 };
    try writer.writeAll(std.mem.sliceAsBytes(&f16_values));

    var result = try loadTensors(std.testing.allocator, buf.items, device, .{});
    defer result.deinit(device);

    const t = result.get("bias").?;
    try std.testing.expectEqual(parser.GGMLType.f16, t.original_dtype);

    const data = t.array.get_data();
    try std.testing.expectEqual(@as(usize, 4), data.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data[0], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data[1], 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data[3], 1e-3);
}

test "loader/load q8_0 tensor" {
    const device_mod = @import("../../device.zig");
    var host = device_mod.HostDevice.init();
    const device = host.reference();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header
    try writer.writeInt(u32, parser.GGUF_MAGIC, .little);
    try writer.writeInt(u32, parser.GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 1, .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor info: q8_0 [32]
    try writeString(writer, "quant_weight");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 32, .little);
    try writer.writeInt(u32, @intFromEnum(parser.GGMLType.q8_0), .little);
    try writer.writeInt(u64, 0, .little);

    padToAlignment(&buf, std.testing.allocator, 32);

    // Q8_0 block: scale=1.0 (f16), 32 quant values all = 5
    const scale_f16: f16 = 1.0;
    try writer.writeAll(&@as([2]u8, @bitCast(scale_f16)));
    for (0..32) |_| {
        try writer.writeByte(@bitCast(@as(i8, 5)));
    }

    var result = try loadTensors(std.testing.allocator, buf.items, device, .{});
    defer result.deinit(device);

    const t = result.get("quant_weight").?;
    try std.testing.expectEqual(parser.GGMLType.q8_0, t.original_dtype);

    const data = t.array.get_data();
    try std.testing.expectEqual(@as(usize, 32), data.len);
    // All values should be 5.0 * 1.0 = 5.0
    for (data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 5.0), v, 1e-3);
    }
}

test "loader/filter tensors" {
    const device_mod = @import("../../device.zig");
    var host = device_mod.HostDevice.init();
    const device = host.reference();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header: 2 tensors, 0 metadata
    try writer.writeInt(u32, parser.GGUF_MAGIC, .little);
    try writer.writeInt(u32, parser.GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 2, .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor 0: "keep_me" f32 [2]
    try writeString(writer, "keep_me");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 2, .little);
    try writer.writeInt(u32, @intFromEnum(parser.GGMLType.f32), .little);
    try writer.writeInt(u64, 0, .little);

    // Tensor 1: "skip_me" f32 [2], offset=8
    try writeString(writer, "skip_me");
    try writer.writeInt(u32, 1, .little);
    try writer.writeInt(u64, 2, .little);
    try writer.writeInt(u32, @intFromEnum(parser.GGMLType.f32), .little);
    try writer.writeInt(u64, 8, .little);

    padToAlignment(&buf, std.testing.allocator, 32);

    // 4 f32 values total
    const tensor_values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try writer.writeAll(std.mem.sliceAsBytes(&tensor_values));

    const filter = [_][]const u8{"keep_me"};
    var result = try loadTensors(std.testing.allocator, buf.items, device, .{
        .filter = &filter,
    });
    defer result.deinit(device);

    try std.testing.expectEqual(@as(usize, 1), result.count());
    try std.testing.expect(result.get("keep_me") != null);
    try std.testing.expect(result.get("skip_me") == null);
}

test "loader/metadata access" {
    const device_mod = @import("../../device.zig");
    var host = device_mod.HostDevice.init();
    const device = host.reference();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    const writer = buf.writer(std.testing.allocator);

    // Header: 0 tensors, 2 metadata
    try writer.writeInt(u32, parser.GGUF_MAGIC, .little);
    try writer.writeInt(u32, parser.GGUF_VERSION_3, .little);
    try writer.writeInt(u64, 0, .little);
    try writer.writeInt(u64, 2, .little);

    // KV: "general.architecture" = "llama"
    try writeString(writer, "general.architecture");
    try writer.writeInt(u32, @intFromEnum(parser.MetadataValueType.string), .little);
    try writeString(writer, "llama");

    // KV: "llama.block_count" = 32
    try writeString(writer, "llama.block_count");
    try writer.writeInt(u32, @intFromEnum(parser.MetadataValueType.uint32), .little);
    try writer.writeInt(u32, 32, .little);

    var result = try loadTensors(std.testing.allocator, buf.items, device, .{});
    defer result.deinit(device);

    const arch = result.getMetadata("general.architecture");
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("llama", arch.?.string);

    const blocks = result.getMetadata("llama.block_count");
    try std.testing.expect(blocks != null);
    try std.testing.expectEqual(@as(u32, 32), blocks.?.uint32);
}

// ── Test helpers ────────────────────────────────────────────────────

fn writeString(writer: anytype, s: []const u8) !void {
    try writer.writeInt(u64, s.len, .little);
    try writer.writeAll(s);
}

fn padToAlignment(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, alignment: usize) void {
    const current = buf.items.len;
    const aligned = ((current + alignment - 1) / alignment) * alignment;
    const pad = aligned - current;
    for (0..pad) |_| {
        buf.append(allocator, 0) catch break;
    }
}
