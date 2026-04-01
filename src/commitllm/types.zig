//! Core types for verifier keys, traces, and verification results.
//!
//! The protocol has two roles:
//! - Prover: runs inference, records intermediates, builds Merkle commitments
//! - Verifier: generates secret r vectors, checks traces via Freivalds

const std = @import("std");
const constants = @import("constants.zig");
const field = @import("field.zig");
const merkle = @import("merkle.zig");

pub const Fp = field.Fp;
pub const ModelConfig = constants.ModelConfig;
pub const MatrixType = constants.MatrixType;
pub const Hash = merkle.Hash;

/// Verifier-secret key material. Contains random vectors r_j and
/// precomputed v_j = r_j^T W_j. Must never be shared with the prover.
pub const VerifierKey = struct {
    config: ModelConfig,
    seed: [32]u8,
    /// Per-matrix-type random vectors r_j (indexed by MatrixType ordinal).
    r_vectors: [][]Fp,
    /// Precomputed v = r^T W per layer per matrix type.
    /// v_vectors[layer][matrix_type_index] = v vector.
    v_vectors: [][][]Fp,
    /// Optional: precomputed v for lm_head.
    v_lm_head: ?[]Fp,
    /// SHA-256 hash of all INT8 weights in canonical order.
    weight_hash: ?Hash,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *VerifierKey) void {
        if (self.v_lm_head) |v| self.allocator.free(v);
        for (self.v_vectors) |layer_vs| {
            for (layer_vs) |v| self.allocator.free(v);
            self.allocator.free(layer_vs);
        }
        self.allocator.free(self.v_vectors);
        for (self.r_vectors) |r| self.allocator.free(r);
        self.allocator.free(self.r_vectors);
        self.* = undefined;
    }

    /// Get the r vector for a given matrix type.
    pub fn rFor(self: *const VerifierKey, mt: MatrixType) []const Fp {
        return self.r_vectors[@intFromEnum(mt)];
    }
};

/// Per-layer trace data from a forward pass (toy model).
pub const LayerTrace = struct {
    x_attn: []i8,
    q: []i32,
    k: []i32,
    v: []i32,
    a: []i8,
    attn_out: []i32,
    x_ffn: []i8,
    g: []i32,
    u: []i32,
    h: []i8,
    ffn_out: []i32,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LayerTrace) void {
        self.allocator.free(self.x_attn);
        self.allocator.free(self.q);
        self.allocator.free(self.k);
        self.allocator.free(self.v);
        self.allocator.free(self.a);
        self.allocator.free(self.attn_out);
        self.allocator.free(self.x_ffn);
        self.allocator.free(self.g);
        self.allocator.free(self.u);
        self.allocator.free(self.h);
        self.allocator.free(self.ffn_out);
        self.* = undefined;
    }
};

/// Retained per-layer state for the Merkle commitment (minimal for toy model).
pub const RetainedLayerState = struct {
    a: []i8,
    scale_a: f32,
};

/// Retained per-token state: one RetainedLayerState per layer.
pub const RetainedTokenState = struct {
    layers: []RetainedLayerState,
};

/// Commitment published by the prover after a generation run.
pub const BatchCommitment = struct {
    merkle_root: Hash,
    io_root: Hash,
    n_tokens: usize,
    weight_hash: ?Hash,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *BatchCommitment) void {
        _ = self;
    }
};
