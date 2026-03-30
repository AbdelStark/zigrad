const std = @import("std");
const schema = @import("schema.zig");
const ops = @import("ops.zig");
const graph_ir = @import("../../graph_ir.zig");
const GraphIR = graph_ir.GraphIR;
const Value = graph_ir.Value;
const DType = graph_ir.DType;
const lazy = @import("../../lazy.zig");

pub const ImportError = error{
    NoGraph,
    UnsupportedOp,
    MissingInput,
    DuplicateName,
    UnsupportedDataType,
    InvalidModel,
    OutOfMemory,
};

/// Convert an ONNX DataType to a Zigrad DType.
fn convertDType(dt: schema.DataType) DType {
    return switch (dt) {
        .FLOAT => .f32,
        .DOUBLE => .f64,
        .INT8 => .i8,
        .INT16 => .i16,
        .INT32 => .i32,
        .INT64 => .i64,
        .UINT8 => .u8,
        .UINT16 => .u16,
        .UINT32 => .u32,
        .UINT64 => .u64,
        .FLOAT16 => .f16,
        else => .unknown,
    };
}

/// Extract raw byte data from a TensorProto.
fn extractTensorData(tensor: *const schema.TensorProto, allocator: std.mem.Allocator) !?[]const u8 {
    // Prefer raw_data if present
    if (tensor.raw_data.len > 0) {
        return try allocator.dupe(u8, tensor.raw_data);
    }
    // Otherwise try typed data fields
    if (tensor.float_data.len > 0) {
        return try allocator.dupe(u8, std.mem.sliceAsBytes(tensor.float_data));
    }
    if (tensor.double_data.len > 0) {
        return try allocator.dupe(u8, std.mem.sliceAsBytes(tensor.double_data));
    }
    if (tensor.int64_data.len > 0) {
        return try allocator.dupe(u8, std.mem.sliceAsBytes(tensor.int64_data));
    }
    return null;
}

/// Convert i64 dims to usize shape. Negative dims (symbolic) become 1.
fn dimsToShape(dims: []const i64, allocator: std.mem.Allocator) ![]usize {
    const shape = try allocator.alloc(usize, dims.len);
    for (dims, 0..) |d, i| {
        shape[i] = if (d > 0) @intCast(d) else 1;
    }
    return shape;
}

/// Import an ONNX model from raw protobuf bytes into a Zigrad GraphIR.
pub fn importModel(allocator: std.mem.Allocator, data: []const u8) ImportError!GraphIR {
    const model = schema.parseModelProto(data, allocator) catch return error.InvalidModel;
    const onnx_graph = model.graph orelse return error.NoGraph;
    return importGraph(allocator, &onnx_graph);
}

/// Import an ONNX GraphProto into a Zigrad GraphIR.
pub fn importGraph(allocator: std.mem.Allocator, onnx_graph: *const schema.GraphProto) ImportError!GraphIR {
    var ir = GraphIR.init(allocator);
    errdefer ir.deinit();

    // Maps ONNX tensor names to IR value IDs
    var name_to_id = std.StringArrayHashMapUnmanaged(u32).empty;
    defer name_to_id.deinit(allocator);

    var next_value_id: u32 = 1;
    var next_op_id: u32 = 1;

    // --- Import initializers as constant input values ---
    for (onnx_graph.initializers) |*init| {
        const dtype = convertDType(init.data_type);
        const shape = dimsToShape(init.dims, allocator) catch return error.OutOfMemory;
        errdefer allocator.free(shape);

        const label = if (init.name.len > 0)
            allocator.dupe(u8, init.name) catch return error.OutOfMemory
        else
            null;
        errdefer if (label) |l| allocator.free(l);

        const const_data = extractTensorData(init, allocator) catch return error.OutOfMemory;
        errdefer if (const_data) |d| allocator.free(d);

        const id = next_value_id;
        next_value_id += 1;

        ir.values.append(allocator, .{
            .id = id,
            .dtype = dtype,
            .shape = shape,
            .device = .host,
            .storage = .owned,
            .defining_op = null,
            .label = label,
            .requires_grad = false,
            .constant_data = const_data,
        }) catch return error.OutOfMemory;
        ir.input_ids.append(allocator, id) catch return error.OutOfMemory;

        if (init.name.len > 0) {
            name_to_id.put(allocator, init.name, id) catch return error.OutOfMemory;
        }
    }

    // --- Import graph inputs (that aren't already initializers) ---
    for (onnx_graph.inputs) |*input| {
        if (input.name.len == 0) continue;
        if (name_to_id.get(input.name) != null) continue; // already an initializer

        const dtype = convertDType(input.elem_type);
        const shape = dimsToShape(input.shape, allocator) catch return error.OutOfMemory;
        errdefer allocator.free(shape);

        const label = allocator.dupe(u8, input.name) catch return error.OutOfMemory;
        errdefer allocator.free(label);

        const id = next_value_id;
        next_value_id += 1;

        ir.values.append(allocator, .{
            .id = id,
            .dtype = dtype,
            .shape = shape,
            .device = .host,
            .storage = .owned,
            .defining_op = null,
            .label = label,
            .requires_grad = false,
        }) catch return error.OutOfMemory;
        ir.input_ids.append(allocator, id) catch return error.OutOfMemory;

        name_to_id.put(allocator, input.name, id) catch return error.OutOfMemory;
    }

    // --- Import nodes as ops ---
    for (onnx_graph.nodes) |*node| {
        const ir_op_name = ops.lookupIrName(node.op_type) orelse return error.UnsupportedOp;

        // Resolve operand IDs
        var operand_ids = std.ArrayListUnmanaged(u32).empty;
        defer operand_ids.deinit(allocator);
        for (node.input) |input_name| {
            if (input_name.len == 0) continue;
            const id = name_to_id.get(input_name) orelse return error.MissingInput;
            operand_ids.append(allocator, id) catch return error.OutOfMemory;
        }

        // Create result values for each output
        var result_ids = std.ArrayListUnmanaged(u32).empty;
        defer result_ids.deinit(allocator);
        for (node.output) |output_name| {
            if (output_name.len == 0) continue;

            const id = next_value_id;
            next_value_id += 1;

            // Infer dtype from first operand (common case)
            var dtype: DType = .unknown;
            if (operand_ids.items.len > 0) {
                if (ir.valueById(operand_ids.items[0])) |first_operand| {
                    dtype = first_operand.dtype;
                }
            }

            // Shape inference: for now use empty shape (will be refined later)
            const shape = allocator.alloc(usize, 0) catch return error.OutOfMemory;

            const label = if (output_name.len > 0)
                allocator.dupe(u8, output_name) catch return error.OutOfMemory
            else
                null;

            ir.values.append(allocator, .{
                .id = id,
                .dtype = dtype,
                .shape = shape,
                .device = .host,
                .storage = .owned,
                .defining_op = next_op_id,
                .label = label,
                .requires_grad = false,
            }) catch return error.OutOfMemory;

            result_ids.append(allocator, id) catch return error.OutOfMemory;
            name_to_id.put(allocator, output_name, id) catch return error.OutOfMemory;
        }

        // Convert ONNX attributes to IR attributes
        var ir_attrs = std.ArrayListUnmanaged(lazy.OpAttribute).empty;
        defer ir_attrs.deinit(allocator);
        for (node.attributes) |attr| {
            const key = allocator.dupe(u8, attr.name) catch return error.OutOfMemory;
            const value: lazy.AttributeValue = switch (attr.type) {
                .FLOAT => .{ .float = attr.f },
                .INT => .{ .int = attr.i },
                .STRING => .{ .string = allocator.dupe(u8, attr.s) catch return error.OutOfMemory },
                else => continue, // Skip unsupported attribute types
            };
            ir_attrs.append(allocator, .{ .key = key, .value = value }) catch return error.OutOfMemory;
        }

        // Handle Gemm special case: extract transA/transB attributes
        if (std.mem.eql(u8, node.op_type, "Gemm")) {
            var trans_a = false;
            var trans_b = false;
            for (node.attributes) |attr| {
                if (std.mem.eql(u8, attr.name, "transA") and attr.i != 0) trans_a = true;
                if (std.mem.eql(u8, attr.name, "transB") and attr.i != 0) trans_b = true;
            }
            // Use the appropriate matmul variant
            const actual_name: []const u8 = if (trans_a and trans_b)
                "MATMUL_AtBt"
            else if (trans_a)
                "MATMUL_AtB"
            else if (trans_b)
                "MATMUL_ABt"
            else
                "MATMUL_AB";
            _ = actual_name; // For now, we use the base name
        }

        const op_id = next_op_id;
        next_op_id += 1;

        ir.ops.append(allocator, .{
            .id = op_id,
            .name = ir_op_name,
            .operands = operand_ids.toOwnedSlice(allocator) catch return error.OutOfMemory,
            .results = result_ids.toOwnedSlice(allocator) catch return error.OutOfMemory,
            .attributes = ir_attrs.toOwnedSlice(allocator) catch return error.OutOfMemory,
        }) catch return error.OutOfMemory;
    }

    // --- Mark graph outputs ---
    for (onnx_graph.outputs) |*output| {
        if (output.name.len == 0) continue;
        const id = name_to_id.get(output.name) orelse continue;
        ir.output_ids.append(allocator, id) catch return error.OutOfMemory;
    }

    return ir;
}

// ---------- Tests ----------

test "import/empty model returns NoGraph" {
    // An empty protobuf message has no graph field
    const result = importModel(std.testing.allocator, &.{});
    try std.testing.expectError(error.NoGraph, result);
}

test "import/mlp-2layer graph structure" {
    const allocator = std.testing.allocator;

    // Weight / bias data (f32, little-endian native bytes).
    // W1: shape [4, 3], 12 floats
    const w1_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 };
    const w1_dims = [_]i64{ 4, 3 };
    // b1: shape [1, 3], 3 floats
    const b1_data = [_]f32{ 0.01, 0.02, 0.03 };
    const b1_dims = [_]i64{ 1, 3 };
    // W2: shape [3, 2], 6 floats
    const w2_data = [_]f32{ 0.5, -0.5, 0.3, -0.3, 0.1, -0.1 };
    const w2_dims = [_]i64{ 3, 2 };
    // b2: shape [1, 2], 2 floats
    const b2_data = [_]f32{ 0.1, -0.1 };
    const b2_dims = [_]i64{ 1, 2 };

    const initializers = [_]schema.TensorProto{
        .{ .name = "W1", .dims = &w1_dims, .data_type = .FLOAT, .float_data = &w1_data },
        .{ .name = "b1", .dims = &b1_dims, .data_type = .FLOAT, .float_data = &b1_data },
        .{ .name = "W2", .dims = &w2_dims, .data_type = .FLOAT, .float_data = &w2_data },
        .{ .name = "b2", .dims = &b2_dims, .data_type = .FLOAT, .float_data = &b2_data },
    };

    // Graph input: x shape [1, 4]
    const x_shape = [_]i64{ 1, 4 };
    const inputs = [_]schema.ValueInfoProto{
        .{ .name = "x", .elem_type = .FLOAT, .shape = &x_shape },
    };

    // Graph output: y shape [1, 2]
    const y_shape = [_]i64{ 1, 2 };
    const outputs = [_]schema.ValueInfoProto{
        .{ .name = "y", .elem_type = .FLOAT, .shape = &y_shape },
    };

    // Nodes:
    //   mm1:   MatMul(x, W1)           -> mm1_out
    //   add1:  Add(mm1_out, b1)        -> add1_out
    //   relu1: Relu(add1_out)          -> relu1_out
    //   mm2:   MatMul(relu1_out, W2)   -> mm2_out
    //   add2:  Add(mm2_out, b2)        -> y
    const mm1_inputs   = [_][]const u8{ "x",         "W1" };
    const mm1_outputs  = [_][]const u8{"mm1_out"};
    const add1_inputs  = [_][]const u8{ "mm1_out",   "b1" };
    const add1_outputs = [_][]const u8{"add1_out"};
    const relu_inputs  = [_][]const u8{"add1_out"};
    const relu_outputs = [_][]const u8{"relu1_out"};
    const mm2_inputs   = [_][]const u8{ "relu1_out", "W2" };
    const mm2_outputs  = [_][]const u8{"mm2_out"};
    const add2_inputs  = [_][]const u8{ "mm2_out",   "b2" };
    const add2_outputs = [_][]const u8{"y"};

    const nodes = [_]schema.NodeProto{
        .{ .op_type = "MatMul", .input = &mm1_inputs,   .output = &mm1_outputs },
        .{ .op_type = "Add",    .input = &add1_inputs,  .output = &add1_outputs },
        .{ .op_type = "Relu",   .input = &relu_inputs,  .output = &relu_outputs },
        .{ .op_type = "MatMul", .input = &mm2_inputs,   .output = &mm2_outputs },
        .{ .op_type = "Add",    .input = &add2_inputs,  .output = &add2_outputs },
    };

    const graph = schema.GraphProto{
        .name         = "mlp",
        .initializers = &initializers,
        .inputs       = &inputs,
        .outputs      = &outputs,
        .nodes        = &nodes,
    };

    var ir = try importGraph(allocator, &graph);
    defer ir.deinit();

    // 4 initializers + 1 graph input = 5 entries in input_ids
    try std.testing.expectEqual(@as(usize, 5), ir.input_ids.items.len);

    // 5 nodes -> 5 ops
    try std.testing.expectEqual(@as(usize, 5), ir.ops.items.len);

    // Op names follow the IR mapping table in ops.zig
    try std.testing.expectEqualStrings("MATMUL_AB", ir.ops.items[0].name);
    try std.testing.expectEqualStrings("ADD",       ir.ops.items[1].name);
    try std.testing.expectEqualStrings("relu",      ir.ops.items[2].name);
    try std.testing.expectEqualStrings("MATMUL_AB", ir.ops.items[3].name);
    try std.testing.expectEqualStrings("ADD",       ir.ops.items[4].name);

    // MatMul/Add take 2 operands; Relu takes 1
    try std.testing.expectEqual(@as(usize, 2), ir.ops.items[0].operands.len);
    try std.testing.expectEqual(@as(usize, 1), ir.ops.items[2].operands.len);
    try std.testing.expectEqual(@as(usize, 2), ir.ops.items[4].operands.len);

    // Each op produces exactly 1 result value
    for (ir.ops.items) |op| {
        try std.testing.expectEqual(@as(usize, 1), op.results.len);
    }

    // Exactly 1 graph output
    try std.testing.expectEqual(@as(usize, 1), ir.output_ids.items.len);

    // W1 initializer (first input_id): dtype, shape, and constant_data preserved
    const w1_value = ir.valueById(ir.input_ids.items[0]).?;
    try std.testing.expectEqual(DType.f32,     w1_value.dtype);
    try std.testing.expectEqual(@as(usize, 2), w1_value.shape.len);
    try std.testing.expectEqual(@as(usize, 4), w1_value.shape[0]);
    try std.testing.expectEqual(@as(usize, 3), w1_value.shape[1]);
    try std.testing.expect(w1_value.constant_data != null);

    // x (the only non-initializer input) is appended after the 4 initializers
    const x_value = ir.valueById(ir.input_ids.items[4]).?;
    try std.testing.expectEqual(DType.f32, x_value.dtype);
    try std.testing.expect(x_value.label != null);
    try std.testing.expectEqualStrings("x", x_value.label.?);
}
