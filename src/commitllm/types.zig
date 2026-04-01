//! Core types for verifier keys, traces, and verification results.
//!
//! The protocol has two roles with a strict information boundary:
//!
//! **Prover** — runs inference, records intermediates, builds Merkle
//! commitments, opens challenged traces. Needs only public model weights.
//!
//! **Verifier** — generates secret r_j, precomputes v_j = r_j^T W_j,
//! checks opened traces using Freivalds + Merkle proofs.

const std = @import("std");
const constants = @import("constants.zig");
const field = @import("field.zig");
const merkle = @import("merkle.zig");

pub const Fp = field.Fp;
pub const ModelConfig = constants.ModelConfig;
pub const MatrixType = constants.MatrixType;
pub const Hash = merkle.Hash;

// ═══════════════════════════════════════════════════════════════════════
// Verification Profile
// ═══════════════════════════════════════════════════════════════════════

/// Family-specific validated parameters. Each supported model family
/// gets a profile with empirically validated tolerances.
pub const VerificationProfile = struct {
    name: []const u8,
    model_family: []const u8,
    /// Max L-inf for bridge x_attn check-and-gate (i8 space).
    bridge_tolerance: u8,
    /// Max L-inf for attention replay comparison (i8 space).
    attention_tolerance: u8,
    /// Max context length for which the corridor bound is validated.
    max_validated_context: u32,
    requires_score_anchoring: bool,

    pub fn qwenW8a8() VerificationProfile {
        return .{
            .name = "qwen-w8a8",
            .model_family = "qwen2",
            .bridge_tolerance = 1,
            .attention_tolerance = 10,
            .max_validated_context = 1164,
            .requires_score_anchoring = false,
        };
    }

    pub fn llamaW8a8() VerificationProfile {
        return .{
            .name = "llama-w8a8",
            .model_family = "llama",
            .bridge_tolerance = 1,
            .attention_tolerance = 10,
            .max_validated_context = 1165,
            .requires_score_anchoring = false,
        };
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Verifier Key
// ═══════════════════════════════════════════════════════════════════════

/// Verifier-secret key material. Contains random vectors r_j and
/// precomputed v_j = r_j^T W_j. Must never be shared with the prover.
pub const VerifierKey = struct {
    version: u32 = 1,
    config: ModelConfig,
    seed: [32]u8,
    source_dtype: []const u8 = "I8",
    /// Per-matrix-type random vectors r_j (indexed by MatrixType ordinal).
    r_vectors: [][]Fp,
    /// Precomputed v = r^T W per layer per matrix type.
    /// v_vectors[layer][matrix_type_index] = v vector.
    v_vectors: [][][]Fp,
    /// Precomputed v for lm_head (optional).
    v_lm_head: ?[]Fp = null,
    /// LM-head weight matrix (optional, for logit binding).
    lm_head: ?[]i8 = null,
    /// SHA-256 of all INT8 weights in canonical order.
    weight_hash: ?Hash = null,
    /// W_o L-inf norms per layer (for Level B margin checks).
    wo_norms: []f32 = &.{},
    /// Max L-inf norm of W_v across all layers.
    max_v_norm: f32 = 0.0,
    /// RMSNorm attention weights per layer.
    rmsnorm_attn_weights: [][]f32 = &.{},
    /// RMSNorm FFN weights per layer.
    rmsnorm_ffn_weights: [][]f32 = &.{},
    /// Final layer norm weights.
    final_norm_weights: ?[]f32 = null,
    /// Per-layer uniform weight scales.
    weight_scales: []f32 = &.{},
    /// Per-layer per-channel weight scales.
    per_channel_weight_scales: [][][]f32 = &.{},
    rmsnorm_eps: f32 = 1e-5,
    /// Embedding table Merkle root.
    embedding_merkle_root: ?Hash = null,
    /// RoPE config hash.
    rope_config_hash: ?Hash = null,
    /// Quantization family: "W8A8", "Q8_0", etc.
    quant_family: ?[]const u8 = null,
    /// Q8_0 block size (32 for standard Q8_0).
    quant_block_size: ?usize = null,
    /// Whether RoPE-aware attention replay is enabled.
    rope_aware_replay: bool = false,
    /// QKV biases per layer (for models with bias terms).
    qkv_biases: [][]f32 = &.{},
    /// Verification profile (family-specific tolerances).
    verification_profile: ?VerificationProfile = null,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *VerifierKey) void {
        if (self.v_lm_head) |v| self.allocator.free(v);
        if (self.lm_head) |l| self.allocator.free(l);
        if (self.final_norm_weights) |f| self.allocator.free(f);
        for (self.v_vectors) |layer_vs| {
            for (layer_vs) |v| self.allocator.free(v);
            self.allocator.free(layer_vs);
        }
        self.allocator.free(self.v_vectors);
        for (self.r_vectors) |r| self.allocator.free(r);
        self.allocator.free(self.r_vectors);
        // Free production-path fields if populated.
        if (self.wo_norms.len > 0) self.allocator.free(self.wo_norms);
        if (self.weight_scales.len > 0) self.allocator.free(self.weight_scales);
        for (self.rmsnorm_attn_weights) |w| self.allocator.free(w);
        if (self.rmsnorm_attn_weights.len > 0) self.allocator.free(self.rmsnorm_attn_weights);
        for (self.rmsnorm_ffn_weights) |w| self.allocator.free(w);
        if (self.rmsnorm_ffn_weights.len > 0) self.allocator.free(self.rmsnorm_ffn_weights);
        for (self.per_channel_weight_scales) |layer| {
            for (layer) |ch| self.allocator.free(ch);
            self.allocator.free(layer);
        }
        if (self.per_channel_weight_scales.len > 0) self.allocator.free(self.per_channel_weight_scales);
        for (self.qkv_biases) |b| self.allocator.free(b);
        if (self.qkv_biases.len > 0) self.allocator.free(self.qkv_biases);
        self.* = undefined;
    }

    /// Get the r vector for a given matrix type.
    pub fn rFor(self: *const VerifierKey, mt: MatrixType) []const Fp {
        return self.r_vectors[@intFromEnum(mt)];
    }

    /// Bridge tolerance from profile, or default 1.
    pub fn bridgeTolerance(self: *const VerifierKey) u8 {
        return if (self.verification_profile) |p| p.bridge_tolerance else 1;
    }

    /// Attention tolerance from profile, or default 10.
    pub fn attentionTolerance(self: *const VerifierKey) u8 {
        return if (self.verification_profile) |p| p.attention_tolerance else 10;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Trace types
// ═══════════════════════════════════════════════════════════════════════

/// Per-layer trace data from a forward pass.
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
    /// Optional: KV cache keys for this layer across all positions.
    kv_cache_k: ?[][]i8 = null,
    /// Optional: KV cache values.
    kv_cache_v: ?[][]i8 = null,
    /// Quantization scales (production W8A8 path).
    scale_x_attn: ?f32 = null,
    scale_a: ?f32 = null,
    scale_x_ffn: ?f32 = null,
    scale_h: ?f32 = null,
    /// Pre-normalization residual (for bridge verification).
    residual: ?[]f32 = null,
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
        if (self.residual) |r| self.allocator.free(r);
        if (self.kv_cache_k) |kk| {
            for (kk) |entry| self.allocator.free(entry);
            self.allocator.free(kk);
        }
        if (self.kv_cache_v) |kv| {
            for (kv) |entry| self.allocator.free(entry);
            self.allocator.free(kv);
        }
        self.* = undefined;
    }
};

/// Retained per-layer state for the Merkle commitment.
/// This is the minimal boundary state retained online (not the full trace).
pub const RetainedLayerState = struct {
    /// Post-attention output (W_o input). Irreducible: depends on full KV prefix via softmax.
    a: []const i8,
    scale_a: f32,
    /// Optional committed bridge trust boundary.
    x_attn_i8: ?[]const i8 = null,
    scale_x_attn: ?f32 = null,
};

/// Retained per-token state: one RetainedLayerState per layer.
pub const RetainedTokenState = struct {
    layers: []const RetainedLayerState,
};

/// KV entry at a single (layer, position) pair.
pub const KvEntry = struct {
    /// Post-RoPE K in f64 (exact values for attention replay).
    k_roped: []const f64,
    /// Dequantized V in f64.
    v_deq: []const f64,
};

// ═══════════════════════════════════════════════════════════════════════
// Commitment types
// ═══════════════════════════════════════════════════════════════════════

/// Four deployment specs that bind the commitment to a specific configuration.
pub const InputSpec = struct {
    tokenizer_hash: Hash,
    system_prompt_hash: ?Hash = null,
    chat_template_hash: ?Hash = null,
    bos_eos_policy: ?[]const u8 = null,
    truncation_policy: ?[]const u8 = null,
    special_token_policy: ?[]const u8 = null,
    padding_policy: ?[]const u8 = null,
};

pub const ModelSpec = struct {
    weight_hash: ?Hash = null,
    quant_hash: ?Hash = null,
    rope_config_hash: ?Hash = null,
    rmsnorm_eps: ?f32 = null,
    adapter_hash: ?Hash = null,
};

pub const DecodeSpec = struct {
    sampler_id: ?[]const u8 = null,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    top_p: ?f32 = null,
    rep_penalty: ?f32 = null,
    grammar_hash: ?Hash = null,
};

pub const OutputSpec = struct {
    eos_token_ids: []const u32 = &.{},
    stop_strings_hash: ?Hash = null,
    max_tokens: ?u32 = null,
    min_tokens: ?u32 = null,
    detokenizer_hash: ?Hash = null,
};

/// Deployment manifest: H(H_input || H_model || H_decode || H_output).
pub const DeploymentManifest = struct {
    input_spec_hash: Hash,
    model_spec_hash: Hash,
    decode_spec_hash: Hash,
    output_spec_hash: Hash,
};

/// Commitment published by the prover after a generation run.
pub const BatchCommitment = struct {
    version: u32 = 4,
    /// Retained trace Merkle root.
    merkle_root: Hash,
    /// IO chain Merkle root (splice/reorder resistance).
    io_root: Hash,
    n_tokens: u32,
    /// H("vi-manifest-v4" || H_input || H_model || H_decode || H_output).
    manifest_hash: ?Hash = null,
    /// SHA-256 of the tokenized prompt.
    prompt_hash: ?Hash = null,
    /// H(seed) — seed revealed at audit, verifier checks H(seed) == seed_commitment.
    seed_commitment: ?Hash = null,
    /// Per-layer KV transcript Merkle roots.
    kv_roots: []Hash = &.{},
    /// Model weight identity hash.
    weight_hash: ?Hash = null,
};

/// Shell opening for a single challenged token: all matmul i32 accumulators.
pub const ShellLayerOpening = struct {
    q: []i32,
    k: []i32,
    v: []i32,
    attn_out: []i32,
    g: []i32,
    u: []i32,
    ffn_out: []i32,
    h: []i8,
    x_ffn: []i8,
};

pub const ShellTokenOpening = struct {
    layers: []ShellLayerOpening,
};

/// Audit tier: how deep the verification goes.
pub const AuditTier = enum {
    /// Structural only: Merkle proofs + IO chain.
    structural,
    /// Routine: structural + Freivalds on all 7 matrices.
    routine,
    /// Deep: routine + full shell replay for prefix tokens.
    deep,
};

/// Individual check result.
pub const CheckResult = struct {
    name: []const u8,
    passed: bool,
    detail: ?[]const u8 = null,
};

/// Full verification result across all phases.
pub const VerificationResult = struct {
    passed: bool,
    tier: AuditTier,
    checks: []CheckResult = &.{},
    n_layers_verified: usize = 0,
    n_matrices_verified: usize = 0,
};
