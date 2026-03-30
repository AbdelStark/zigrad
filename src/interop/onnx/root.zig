pub const proto = @import("proto.zig");
pub const schema = @import("schema.zig");
pub const ops = @import("ops.zig");
pub const import_ = @import("import.zig");

/// Import an ONNX model from raw protobuf bytes into a Zigrad GraphIR.
pub const importModel = import_.importModel;

/// Import an ONNX graph (without the model wrapper).
pub const importGraph = import_.importGraph;

test {
    _ = proto;
    _ = schema;
    _ = ops;
    _ = import_;
}
