pub const parser = @import("parser.zig");
pub const quant = @import("quant.zig");
pub const loader = @import("loader.zig");

// Top-level re-exports
pub const GGMLType = parser.GGMLType;
pub const GGUFFile = parser.GGUFFile;
pub const TensorInfo = parser.TensorInfo;
pub const MetadataValue = parser.MetadataValue;
pub const MetadataKV = parser.MetadataKV;

pub const TensorMap = loader.TensorMap;
pub const LoadedTensor = loader.LoadedTensor;
pub const LoadOptions = loader.LoadOptions;

/// Parse a raw GGUF file. Returns the parsed structure referencing into `data`.
pub const parse = parser.parse;

/// Load all (or filtered) tensors from a GGUF file into device memory as f32.
pub const loadTensors = loader.loadTensors;

/// Dequantize raw bytes from a GGUF tensor to f32.
pub const dequantize = quant.dequantize;

test {
    _ = parser;
    _ = quant;
    _ = loader;
}
