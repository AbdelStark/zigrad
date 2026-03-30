/// Maps ONNX operator names to Zigrad IR op names.
///
/// ONNX op names follow PascalCase (e.g., "Add", "MatMul", "Relu").
/// Zigrad IR op names follow the convention established by the lazy
/// capture system (e.g., "ADD", "MATMUL_AB", "relu").

pub const OpMapping = struct {
    onnx_name: []const u8,
    ir_name: []const u8,
    /// Number of required inputs.
    min_inputs: u32,
    max_inputs: u32,
};

/// Core op mappings. Covers opset 13+ core ops.
pub const mappings = [_]OpMapping{
    // Elementwise binary
    .{ .onnx_name = "Add", .ir_name = "ADD", .min_inputs = 2, .max_inputs = 2 },
    .{ .onnx_name = "Sub", .ir_name = "SUB", .min_inputs = 2, .max_inputs = 2 },
    .{ .onnx_name = "Mul", .ir_name = "MUL", .min_inputs = 2, .max_inputs = 2 },
    .{ .onnx_name = "Div", .ir_name = "DIV", .min_inputs = 2, .max_inputs = 2 },

    // Activations
    .{ .onnx_name = "Relu", .ir_name = "relu", .min_inputs = 1, .max_inputs = 1 },
    .{ .onnx_name = "Sigmoid", .ir_name = "sigmoid", .min_inputs = 1, .max_inputs = 1 },
    .{ .onnx_name = "Tanh", .ir_name = "tanh", .min_inputs = 1, .max_inputs = 1 },
    .{ .onnx_name = "Exp", .ir_name = "EXP", .min_inputs = 1, .max_inputs = 1 },
    .{ .onnx_name = "Sqrt", .ir_name = "sqrt", .min_inputs = 1, .max_inputs = 1 },

    // Linear algebra
    .{ .onnx_name = "MatMul", .ir_name = "MATMUL_AB", .min_inputs = 2, .max_inputs = 2 },
    .{ .onnx_name = "Gemm", .ir_name = "MATMUL_AB", .min_inputs = 2, .max_inputs = 3 },

    // Reductions
    .{ .onnx_name = "ReduceSum", .ir_name = "SUM", .min_inputs = 1, .max_inputs = 2 },
    .{ .onnx_name = "ReduceMax", .ir_name = "MAX", .min_inputs = 1, .max_inputs = 2 },

    // Shape
    .{ .onnx_name = "Transpose", .ir_name = "TRANSPOSE", .min_inputs = 1, .max_inputs = 1 },
    .{ .onnx_name = "Identity", .ir_name = "clone", .min_inputs = 1, .max_inputs = 1 },

    // Softmax
    .{ .onnx_name = "Softmax", .ir_name = "softmax", .min_inputs = 1, .max_inputs = 1 },
};

const std = @import("std");

/// Look up the IR op name for a given ONNX op name.
pub fn lookupIrName(onnx_op: []const u8) ?[]const u8 {
    for (mappings) |m| {
        if (std.mem.eql(u8, m.onnx_name, onnx_op)) return m.ir_name;
    }
    return null;
}

/// Check if an ONNX op is supported.
pub fn isSupported(onnx_op: []const u8) bool {
    return lookupIrName(onnx_op) != null;
}

test "ops/lookup known ops" {
    try std.testing.expectEqualStrings("ADD", lookupIrName("Add").?);
    try std.testing.expectEqualStrings("relu", lookupIrName("Relu").?);
    try std.testing.expectEqualStrings("MATMUL_AB", lookupIrName("MatMul").?);
    try std.testing.expect(lookupIrName("UnknownOp") == null);
}
