const std = @import("std");
const proto = @import("proto.zig");

/// ONNX data types (from onnx.proto TensorProto.DataType).
pub const DataType = enum(u32) {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    _,
};

/// An ONNX tensor (initializer or constant).
pub const TensorProto = struct {
    name: []const u8 = "",
    dims: []const i64 = &.{},
    data_type: DataType = .UNDEFINED,
    /// Raw binary data (when data is stored in raw_data field).
    raw_data: []const u8 = "",
    /// Float data (when data is stored as repeated float).
    float_data: []const f32 = &.{},
    /// Double data.
    double_data: []const f64 = &.{},
    /// Int32 data.
    int32_data: []const i32 = &.{},
    /// Int64 data.
    int64_data: []const i64 = &.{},
};

/// An ONNX attribute on a node.
pub const AttributeProto = struct {
    name: []const u8 = "",
    type: AttributeType = .UNDEFINED,
    f: f32 = 0,
    i: i64 = 0,
    s: []const u8 = "",
    t: ?TensorProto = null,
    floats: []const f32 = &.{},
    ints: []const i64 = &.{},
};

pub const AttributeType = enum(u32) {
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
    _,
};

/// An ONNX computation node.
pub const NodeProto = struct {
    input: []const []const u8 = &.{},
    output: []const []const u8 = &.{},
    name: []const u8 = "",
    op_type: []const u8 = "",
    domain: []const u8 = "",
    attributes: []const AttributeProto = &.{},
};

/// Value type info (shape + dtype).
pub const ValueInfoProto = struct {
    name: []const u8 = "",
    elem_type: DataType = .UNDEFINED,
    shape: []const i64 = &.{},
};

/// An ONNX graph (main computation graph).
pub const GraphProto = struct {
    name: []const u8 = "",
    nodes: []const NodeProto = &.{},
    initializers: []const TensorProto = &.{},
    inputs: []const ValueInfoProto = &.{},
    outputs: []const ValueInfoProto = &.{},
};

/// An ONNX opset import.
pub const OpsetImport = struct {
    domain: []const u8 = "",
    version: i64 = 0,
};

/// Top-level ONNX model.
pub const ModelProto = struct {
    ir_version: i64 = 0,
    opset_imports: []const OpsetImport = &.{},
    producer_name: []const u8 = "",
    producer_version: []const u8 = "",
    domain: []const u8 = "",
    model_version: i64 = 0,
    doc_string: []const u8 = "",
    graph: ?GraphProto = null,
};

// ---------- Parsing ----------

pub const ParseError = proto.DecodeError || std.mem.Allocator.Error;

/// Parse a TensorProto from protobuf bytes.
pub fn parseTensorProto(data: []const u8, allocator: std.mem.Allocator) ParseError!TensorProto {
    var result = TensorProto{};
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // dims (repeated int64, packed)
                if (field.wire_type == .length_delimited) {
                    result.dims = try proto.readPackedVarint64(field.data.bytes, allocator);
                }
            },
            2 => { // data_type
                if (field.wire_type == .varint)
                    result.data_type = @enumFromInt(@as(u32, @intCast(field.data.varint)));
            },
            4 => { // float_data (packed)
                if (field.wire_type == .length_delimited) {
                    result.float_data = try proto.readPackedF32(field.data.bytes, allocator);
                }
            },
            5 => { // double_data (packed)
                if (field.wire_type == .length_delimited) {
                    result.double_data = try proto.readPackedF64(field.data.bytes, allocator);
                }
            },
            7 => { // int64_data (packed)
                if (field.wire_type == .length_delimited) {
                    result.int64_data = try proto.readPackedVarint64(field.data.bytes, allocator);
                }
            },
            8 => { // name
                if (field.wire_type == .length_delimited)
                    result.name = field.data.bytes;
            },
            13 => { // raw_data
                if (field.wire_type == .length_delimited)
                    result.raw_data = field.data.bytes;
            },
            else => {},
        }
    }
    return result;
}

/// Parse an AttributeProto from protobuf bytes.
pub fn parseAttributeProto(data: []const u8, allocator: std.mem.Allocator) ParseError!AttributeProto {
    var result = AttributeProto{};
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // name
                if (field.wire_type == .length_delimited)
                    result.name = field.data.bytes;
            },
            2 => { // f (float, stored as fixed32)
                if (field.wire_type == .fixed32)
                    result.f = @bitCast(field.data.fixed32);
            },
            3 => { // i (int64)
                if (field.wire_type == .varint)
                    result.i = @bitCast(field.data.varint);
            },
            4 => { // s (bytes)
                if (field.wire_type == .length_delimited)
                    result.s = field.data.bytes;
            },
            5 => { // t (TensorProto)
                if (field.wire_type == .length_delimited)
                    result.t = try parseTensorProto(field.data.bytes, allocator);
            },
            7 => { // floats (repeated, packed)
                if (field.wire_type == .length_delimited)
                    result.floats = try proto.readPackedF32(field.data.bytes, allocator);
            },
            8 => { // ints (repeated, packed)
                if (field.wire_type == .length_delimited)
                    result.ints = try proto.readPackedVarint64(field.data.bytes, allocator);
            },
            20 => { // type
                if (field.wire_type == .varint)
                    result.type = @enumFromInt(@as(u32, @intCast(field.data.varint)));
            },
            else => {},
        }
    }
    return result;
}

/// Parse a NodeProto from protobuf bytes.
pub fn parseNodeProto(data: []const u8, allocator: std.mem.Allocator) ParseError!NodeProto {
    var inputs = std.ArrayListUnmanaged([]const u8).empty;
    var outputs = std.ArrayListUnmanaged([]const u8).empty;
    var attrs = std.ArrayListUnmanaged(AttributeProto).empty;
    var result = NodeProto{};

    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // input (repeated string)
                if (field.wire_type == .length_delimited)
                    try inputs.append(allocator, field.data.bytes);
            },
            2 => { // output (repeated string)
                if (field.wire_type == .length_delimited)
                    try outputs.append(allocator, field.data.bytes);
            },
            3 => { // name
                if (field.wire_type == .length_delimited)
                    result.name = field.data.bytes;
            },
            4 => { // op_type
                if (field.wire_type == .length_delimited)
                    result.op_type = field.data.bytes;
            },
            5 => { // attributes (repeated)
                if (field.wire_type == .length_delimited)
                    try attrs.append(allocator, try parseAttributeProto(field.data.bytes, allocator));
            },
            7 => { // domain
                if (field.wire_type == .length_delimited)
                    result.domain = field.data.bytes;
            },
            else => {},
        }
    }

    result.input = try inputs.toOwnedSlice(allocator);
    result.output = try outputs.toOwnedSlice(allocator);
    result.attributes = try attrs.toOwnedSlice(allocator);
    return result;
}

/// Parse a ValueInfoProto from protobuf bytes.
pub fn parseValueInfoProto(data: []const u8, allocator: std.mem.Allocator) ParseError!ValueInfoProto {
    var result = ValueInfoProto{};
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // name
                if (field.wire_type == .length_delimited)
                    result.name = field.data.bytes;
            },
            2 => { // type (TypeProto, nested)
                if (field.wire_type == .length_delimited)
                    parseTypeProto(field.data.bytes, &result, allocator) catch {};
            },
            else => {},
        }
    }
    return result;
}

/// Parse TypeProto to extract elem_type and shape.
fn parseTypeProto(data: []const u8, info: *ValueInfoProto, allocator: std.mem.Allocator) !void {
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // tensor_type
                if (field.wire_type == .length_delimited)
                    try parseTensorTypeProto(field.data.bytes, info, allocator);
            },
            else => {},
        }
    }
}

fn parseTensorTypeProto(data: []const u8, info: *ValueInfoProto, allocator: std.mem.Allocator) !void {
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // elem_type
                if (field.wire_type == .varint)
                    info.elem_type = @enumFromInt(@as(u32, @intCast(field.data.varint)));
            },
            2 => { // shape (TensorShapeProto)
                if (field.wire_type == .length_delimited)
                    info.shape = try parseShapeProto(field.data.bytes, allocator);
            },
            else => {},
        }
    }
}

fn parseShapeProto(data: []const u8, allocator: std.mem.Allocator) ![]const i64 {
    var dims = std.ArrayListUnmanaged(i64).empty;
    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // dim (repeated TensorShapeProto.Dimension)
                if (field.wire_type == .length_delimited) {
                    var dim_iter = proto.iterate(field.data.bytes);
                    while (dim_iter.next()) |dim_field| {
                        switch (dim_field.number) {
                            1 => { // dim_value
                                if (dim_field.wire_type == .varint)
                                    try dims.append(allocator, @bitCast(dim_field.data.varint));
                            },
                            else => {},
                        }
                    }
                }
            },
            else => {},
        }
    }
    return dims.toOwnedSlice(allocator);
}

/// Parse a GraphProto from protobuf bytes.
pub fn parseGraphProto(data: []const u8, allocator: std.mem.Allocator) ParseError!GraphProto {
    var nodes = std.ArrayListUnmanaged(NodeProto).empty;
    var initializers = std.ArrayListUnmanaged(TensorProto).empty;
    var inputs = std.ArrayListUnmanaged(ValueInfoProto).empty;
    var outputs = std.ArrayListUnmanaged(ValueInfoProto).empty;
    var result = GraphProto{};

    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // node (repeated)
                if (field.wire_type == .length_delimited)
                    try nodes.append(allocator, try parseNodeProto(field.data.bytes, allocator));
            },
            2 => { // name
                if (field.wire_type == .length_delimited)
                    result.name = field.data.bytes;
            },
            5 => { // initializer (repeated TensorProto)
                if (field.wire_type == .length_delimited)
                    try initializers.append(allocator, try parseTensorProto(field.data.bytes, allocator));
            },
            11 => { // input (repeated ValueInfoProto)
                if (field.wire_type == .length_delimited)
                    try inputs.append(allocator, try parseValueInfoProto(field.data.bytes, allocator));
            },
            12 => { // output (repeated ValueInfoProto)
                if (field.wire_type == .length_delimited)
                    try outputs.append(allocator, try parseValueInfoProto(field.data.bytes, allocator));
            },
            else => {},
        }
    }

    result.nodes = try nodes.toOwnedSlice(allocator);
    result.initializers = try initializers.toOwnedSlice(allocator);
    result.inputs = try inputs.toOwnedSlice(allocator);
    result.outputs = try outputs.toOwnedSlice(allocator);
    return result;
}

/// Parse a ModelProto from protobuf bytes.
pub fn parseModelProto(data: []const u8, allocator: std.mem.Allocator) ParseError!ModelProto {
    var opsets = std.ArrayListUnmanaged(OpsetImport).empty;
    var result = ModelProto{};

    var iter = proto.iterate(data);
    while (iter.next()) |field| {
        switch (field.number) {
            1 => { // ir_version
                if (field.wire_type == .varint)
                    result.ir_version = @bitCast(field.data.varint);
            },
            2 => { // producer_name
                if (field.wire_type == .length_delimited)
                    result.producer_name = field.data.bytes;
            },
            3 => { // producer_version
                if (field.wire_type == .length_delimited)
                    result.producer_version = field.data.bytes;
            },
            4 => { // domain
                if (field.wire_type == .length_delimited)
                    result.domain = field.data.bytes;
            },
            5 => { // model_version
                if (field.wire_type == .varint)
                    result.model_version = @bitCast(field.data.varint);
            },
            6 => { // doc_string
                if (field.wire_type == .length_delimited)
                    result.doc_string = field.data.bytes;
            },
            7 => { // graph
                if (field.wire_type == .length_delimited)
                    result.graph = try parseGraphProto(field.data.bytes, allocator);
            },
            8 => { // opset_import (repeated)
                if (field.wire_type == .length_delimited) {
                    var opset = OpsetImport{};
                    var opset_iter = proto.iterate(field.data.bytes);
                    while (opset_iter.next()) |opset_field| {
                        switch (opset_field.number) {
                            1 => { // domain
                                if (opset_field.wire_type == .length_delimited)
                                    opset.domain = opset_field.data.bytes;
                            },
                            2 => { // version
                                if (opset_field.wire_type == .varint)
                                    opset.version = @bitCast(opset_field.data.varint);
                            },
                            else => {},
                        }
                    }
                    try opsets.append(allocator, opset);
                }
            },
            else => {},
        }
    }

    result.opset_imports = try opsets.toOwnedSlice(allocator);
    return result;
}
