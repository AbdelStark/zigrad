//! Model configurations and matrix type definitions.
//!
//! Each model config encodes the dimensions needed for Freivalds checks:
//! hidden dim, KV dim (under GQA), FFN intermediate dim, number of layers,
//! and head counts.

pub const MatrixType = enum(u3) {
    wq,
    wk,
    wv,
    wo,
    wg,
    wu,
    wd,
    lm_head,

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
            .wq => cfg.hidden_dim,
            .wk => cfg.hidden_dim,
            .wv => cfg.hidden_dim,
            .wo => cfg.hidden_dim,
            .wg => cfg.hidden_dim,
            .wu => cfg.hidden_dim,
            .wd => cfg.ffn_dim,
            .lm_head => cfg.hidden_dim,
        };
    }
};

pub const ModelConfig = struct {
    hidden_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    d_head: usize,
    n_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    rope_theta: f64,

    /// Toy model for end-to-end testing. Must match Rust ModelConfig::toy().
    pub fn toy() ModelConfig {
        return .{
            .hidden_dim = 16,
            .kv_dim = 4, // 2 KV heads * 2
            .ffn_dim = 32,
            .d_head = 2,
            .n_layers = 2,
            .n_q_heads = 8,
            .n_kv_heads = 2,
            .vocab_size = 64,
            .rope_theta = 10000.0,
        };
    }
};

const std = @import("std");

test "matrix_dims_toy" {
    const cfg = ModelConfig.toy();
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wq.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wq.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 4), MatrixType.wk.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wk.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 32), MatrixType.wg.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wg.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 16), MatrixType.wd.outputDim(cfg));
    try std.testing.expectEqual(@as(usize, 32), MatrixType.wd.inputDim(cfg));
    try std.testing.expectEqual(@as(usize, 64), MatrixType.lm_head.outputDim(cfg));
}

test "per_layer_count" {
    try std.testing.expectEqual(@as(usize, 7), MatrixType.PER_LAYER.len);
    try std.testing.expectEqual(@as(usize, 8), MatrixType.ALL.len);
}
