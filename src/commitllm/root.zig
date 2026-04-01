//! CommitLLM: Cryptographic commit-and-audit protocol for INT8 LLM inference.
//!
//! This module implements the core verification primitives from the CommitLLM
//! scheme: three-tier Freivalds algebraic matmul checks (Fp/Fp64/Fp128),
//! SHA-256 Merkle trees with domain-separated hashing, SiLU activation
//! verification (toy + production W8A8 + per-channel paths), Q8_0 block-aware
//! Freivalds, and the complete keygen → forward → commit → verify pipeline.
//!
//! The protocol has two roles:
//! - **Prover**: runs inference with ~12% tracing overhead, returns compact receipts.
//! - **Verifier**: checks receipts on CPU in ~1.3ms per challenged token (Llama 70B).
//!
//! Not a ZK system. Uses Freivalds algebraic fingerprints (v = r^T W precomputed)
//! for O(n) per-matrix verification with false-accept probability ≤ 1/p:
//!   Fp:   ≈ 2.3e-10
//!   Fp64: ≈ 4.3e-19
//!   Fp128: ≈ 5.9e-39

pub const field = @import("field.zig");
pub const freivalds = @import("freivalds.zig");
pub const merkle = @import("merkle.zig");
pub const silu = @import("silu.zig");
pub const requantize = @import("requantize.zig");
pub const constants = @import("constants.zig");
pub const types = @import("types.zig");
pub const toy_model = @import("toy_model.zig");
pub const pipeline = @import("pipeline.zig");
pub const diff_tests = @import("diff_tests.zig");

// Convenience re-exports
pub const Fp = field.Fp;
pub const Fp64 = field.Fp64;
pub const Fp128 = field.Fp128;
pub const U256 = field.U256;
pub const ModelConfig = constants.ModelConfig;
pub const MatrixType = constants.MatrixType;
pub const MerkleTree = merkle.MerkleTree;
pub const VerifierKey = types.VerifierKey;
pub const VerificationProfile = types.VerificationProfile;

test {
    _ = field;
    _ = freivalds;
    _ = merkle;
    _ = silu;
    _ = requantize;
    _ = constants;
    _ = types;
    _ = toy_model;
    _ = pipeline;
    _ = diff_tests;
}
