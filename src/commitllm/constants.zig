//! Model configurations and matrix type definitions.
//!
//! Each model config encodes the dimensions needed for Freivalds checks:
//! hidden dim, KV dim (under GQA), FFN intermediate dim, number of layers,
//! and head counts.

const std = @import("std");

/// Q8_0 block size: 32 quantized int8 values per block, with one f16 scale factor.
pub const Q8_0_BLOCK_SIZE: usize = 32;

pub const MatrixType = enum(u3) {
    wq,
    wk,
    wv,
    wo,
    wg,
    wu,
    wd,
    lm_head, // 8 variants fits exactly in u3 (0..7)

    /// The 7 per-layer matrices (one set per transformer layer).
    pub const PER_LAYER = [_]MatrixType{ .wq, .wk, .wv, .wo, .wg, .wu, .wd };

    /// All 8 matrices including the global LM-head.
    pub const ALL = [_]MatrixType{ .wq, .wk, .wv, .wo, .wg, .wu, .wd, .lm_head };

    /// Output dimension m_j (the dimension of r_j).
    pub fn outputDim(self: MatrixType, cfg: ModelConfig) usize {
        return switch (self) {
            .wq => cfg.hidden_dim,
            .wk => cfg.kv_dim,
            .wv => cfg.kv_dim,
            .wo => cfg.hidden_dim,
            .wg => cfg.ffn_dim,
            .wu => cfg.ffn_dim,
            .wd => cfg.hidden_dim,
            .lm_head => cfg.vocab_size,
        };
    }

    /// Input dimension n_j (the dimension of v_j = r_j^T W_j).
    pub fn inputDim(self: MatrixType, cfg: ModelConfig) usize {
        return switch (self) {
            .wq, .wk, .wv, .wo, .wg, .wu => cfg.hidden_dim,
            .wd => cfg.ffn_dim,
            .lm_head => cfg.hidden_dim,
        };
    }

    /// Safetensors weight name pattern. Layer index substituted for `{}`.
    pub fn weightName(self: MatrixType) []const u8 {
        return switch (self) {
            .wq => "model.layers.{}.self_attn.q_proj.weight",
            .wk => "model.layers.{}.self_attn.k_proj.weight",
            .wv => "model.layers.{}.self_attn.v_proj.weight",
            .wo => "model.layers.{}.self_attn.o_proj.weight",
            .wg => "model.layers.{}.mlp.gate_proj.weight",
            .wu => "model.layers.{}.mlp.up_proj.weight",
            .wd => "model.layers.{}.mlp.down_proj.weight",
            .lm_head => "lm_head.weight",
        };
    }

    /// Safetensors bias name pattern. Only Q/K/V may have bias.
    pub fn biasName(self: MatrixType) ?[]const u8 {
        return switch (self) {
            .wq => "model.layers.{}.self_attn.q_proj.bias",
            .wk => "model.layers.{}.self_attn.k_proj.bias",
            .wv => "model.layers.{}.self_attn.v_proj.bias",
            else => null,
        };
    }

    /// Safetensors per-channel weight scale name pattern (W8A8 models).
    pub fn weightScaleName(self: MatrixType) []const u8 {
        return switch (self) {
            .wq => "model.layers.{}.self_attn.q_proj.weight_scale",
            .wk => "model.layers.{}.self_attn.k_proj.weight_scale",
            .wv => "model.layers.{}.self_attn.v_proj.weight_scale",
            .wo => "model.layers.{}.self_attn.o_proj.weight_scale",
            .wg => "model.layers.{}.mlp.gate_proj.weight_scale",
            .wu => "model.layers.{}.mlp.up_proj.weight_scale",
            .wd => "model.layers.{}.mlp.down_proj.weight_scale",
            .lm_head => "lm_head.weight_scale",
        };
    }
};

/// RoPE scaling configuration for extended-context models (e.g., Llama 3.1).
pub const RopeScaling = struct {
    /// Scaling type: "llama3", "linear", "dynamic".
    rope_type: []const u8,
    /// Position scaling factor (e.g. 8.0 for Llama 3.1).
    factor: f64,
    /// Low frequency factor for band boundary (default 1.0).
    low_freq_factor: f64,
    /// High frequency factor for band boundary (default 4.0).
    high_freq_factor: f64,
    /// Original training context length before scaling (e.g. 8192).
    original_max_position_embeddings: usize,
};

pub const ModelConfig = struct {
    name: []const u8 = "unnamed",
    hidden_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    d_head: usize,
    n_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    rope_theta: f64,
    rope_scaling: ?RopeScaling = null,

    /// Toy model for end-to-end testing. Matches Rust ModelConfig::toy().
    pub fn toy() ModelConfig {
        return .{
            .name = "toy",
            .hidden_dim = 16,
            .kv_dim = 4,
            .ffn_dim = 32,
            .d_head = 2,
            .n_layers = 2,
            .n_q_heads = 8,
            .n_kv_heads = 2,
            .vocab_size = 64,
            .rope_theta = 10000.0,
        };
    }

    pub fn llama70b() ModelConfig {
        return .{
            .name = "Llama-3-70B",
            .hidden_dim = 8192,
            .kv_dim = 1024,
            .ffn_dim = 28672,
            .d_head = 128,
            .n_layers = 80,
            .n_q_heads = 64,
            .n_kv_heads = 8,
            .vocab_size = 128256,
            .rope_theta = 500000.0,
        };
    }

    pub fn llama8b() ModelConfig {
        return .{
            .name = "Llama-3-8B",
            .hidden_dim = 4096,
            .kv_dim = 1024,
            .ffn_dim = 14336,
            .d_head = 128,
            .n_layers = 32,
            .n_q_heads = 32,
            .n_kv_heads = 8,
            .vocab_size = 128256,
            .rope_theta = 500000.0,
        };
    }

    /// Compute scaled inverse frequencies for RoPE.
    /// Returns `d_head / 2` frequency values.
    pub fn scaledInvFreq(self: ModelConfig, allocator: std.mem.Allocator) ![]f64 {
        const half = self.d_head / 2;
        const freqs = try allocator.alloc(f64, half);

        for (0..half) |k| {
            freqs[k] = 1.0 / std.math.pow(f64, self.rope_theta, @as(f64, @floatFromInt(2 * k)) / @as(f64, @floatFromInt(self.d_head)));
        }

        if (self.rope_scaling) |scaling| {
            if (std.mem.eql(u8, scaling.rope_type, "llama3")) {
                const old_ctx: f64 = @floatFromInt(scaling.original_max_position_embeddings);
                const low_freq_wavelen = old_ctx / scaling.low_freq_factor;
                const high_freq_wavelen = old_ctx / scaling.high_freq_factor;

                for (0..half) |k| {
                    const wavelen = 2.0 * std.math.pi / freqs[k];
                    if (wavelen < high_freq_wavelen) {
                        // High frequency: no scaling
                    } else if (wavelen > low_freq_wavelen) {
                        freqs[k] /= scaling.factor;
                    } else {
                        const smooth = (old_ctx / wavelen - scaling.low_freq_factor) /
                            (scaling.high_freq_factor - scaling.low_freq_factor);
                        freqs[k] *= (1.0 - smooth) / scaling.factor + smooth;
                    }
                }
            } else if (std.mem.eql(u8, scaling.rope_type, "linear")) {
                for (0..half) |k| {
                    freqs[k] /= scaling.factor;
                }
            }
        }

        return freqs;
    }
};

test "matrix_dims_toy" {
    const cfg = ModelConfig.toy();
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wq.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wq.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 4), MatrixType.wk.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 32), MatrixType.wg.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wd.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 32), MatrixType.wd.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 64), MatrixType.lm_head.outputDim(cfg));
}

test "matrix_dims_llama70b" {
    const cfg = ModelConfig.llama70b();
    try std.testing.expectEqual(@as(usize, 8192), MatrixType.wq.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 1024), MatrixType.wk.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 28672), MatrixType.wg.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 128256), MatrixType.lm_head.outputDim(cfg));
}

test "per_layer_count" {
    try std.testing.expectEqual(@as(usize, 7), MatrixType.PER_LAYER.len);
    try std.testing.expectEqual(@as(usize, 8), MatrixType.ALL.len);
}

test "scaled_inv_freq_basic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    const freqs = try cfg.scaledInvFreq(allocator);
    defer allocator.free(freqs);
    try std.testing.expectEqual(@as(usize, 1), freqs.len); // d_head=2, half=1
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), freqs[0], 1e-6); // theta^(0/2) = 1.0
}
