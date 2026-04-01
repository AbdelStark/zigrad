//! CommitLLM: Cryptographic commit-and-audit protocol for INT8 LLM inference.
//!
//! This module implements the core verification primitives from the CommitLLM
//! scheme: Freivalds algebraic matmul checks, SHA-256 Merkle trees, SiLU
//! activation verification, and the complete keygen → forward → commit → verify
//! pipeline for toy model testing.
//!
//! The protocol has two roles:
//! - Prover: runs inference with ~12% tracing overhead, returns compact receipts
//! - Verifier: checks receipts on CPU in ~1.3ms per challenged token (Llama 70B)
//!
//! Not a ZK system. Uses Freivalds algebraic fingerprints (r^T W precomputed)
//! for O(n) per-matrix verification with false-accept probability ≤ 1/p ≈ 2.3e-10.

pub const field = @import("field.zig");
pub const freivalds = @import("freivalds.zig");
pub const merkle = @import("merkle.zig");
pub const silu = @import("silu.zig");
pub const requantize = @import("requantize.zig");
pub const constants = @import("constants.zig");
pub const types = @import("types.zig");
pub const toy_model = @import("toy_model.zig");
pub const pipeline = @import("pipeline.zig");

// Convenience re-exports
pub const Fp = field.Fp;
pub const Fp64 = field.Fp64;
pub const ModelConfig = constants.ModelConfig;
pub const MatrixType = constants.MatrixType;
pub const MerkleTree = merkle.MerkleTree;

pub const diff_tests = @import("diff_tests.zig");

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
