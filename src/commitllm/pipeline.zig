//! End-to-end pipeline: keygen → forward → commit → verify.
//!
//! Integration test that proves all building blocks compose correctly.

const std = @import("std");
const field = @import("field.zig");
const constants = @import("constants.zig");
const freivalds = @import("freivalds.zig");
const merkle = @import("merkle.zig");
const types = @import("types.zig");
const toy_model = @import("toy_model.zig");
const requantize_mod = @import("requantize.zig");

const Fp = field.Fp;
const ModelConfig = constants.ModelConfig;
const MatrixType = constants.MatrixType;

/// Result of running the e2e pipeline.
pub const PipelineResult = struct {
    all_freivalds_passed: bool,
    merkle_proofs_valid: bool,
    n_layers_verified: usize,
    n_matrices_verified: usize,
};

/// Run the complete e2e verification pipeline on a toy model.
pub fn runE2E(allocator: std.mem.Allocator) !PipelineResult {
    const cfg = ModelConfig.toy();
    var model = try toy_model.generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try toy_model.generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    // Generate input
    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    var traces = try toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (traces) |*t| t.deinit();
        allocator.free(traces);
    }

    // Phase 1: Freivalds verification
    var all_freivalds_passed = true;
    var n_matrices_verified: usize = 0;

    for (0..cfg.n_layers) |layer| {
        const lt = &traces[layer];
        const layer_vs = key.v_vectors[layer];

        // Check all 7 per-layer matrices
        const checks = [_]struct { v_idx: usize, input: []const i8, r_mt: MatrixType, output: []const i32 }{
            .{ .v_idx = 0, .input = lt.x_attn, .r_mt = .wq, .output = lt.q },
            .{ .v_idx = 1, .input = lt.x_attn, .r_mt = .wk, .output = lt.k },
            .{ .v_idx = 2, .input = lt.x_attn, .r_mt = .wv, .output = lt.v },
            .{ .v_idx = 3, .input = lt.a, .r_mt = .wo, .output = lt.attn_out },
            .{ .v_idx = 4, .input = lt.x_ffn, .r_mt = .wg, .output = lt.g },
            .{ .v_idx = 5, .input = lt.x_ffn, .r_mt = .wu, .output = lt.u },
            .{ .v_idx = 6, .input = lt.h, .r_mt = .wd, .output = lt.ffn_out },
        };

        for (checks) |chk| {
            const passed = freivalds.check(
                layer_vs[chk.v_idx],
                chk.input,
                key.rFor(chk.r_mt),
                chk.output,
            );
            if (!passed) all_freivalds_passed = false;
            n_matrices_verified += 1;
        }
    }

    // Phase 2: Merkle tree commitment and verification
    var merkle_proofs_valid = true;

    // Build retained state leaves for each token (here just one token)
    const retained_layers = try allocator.alloc(merkle.RetainedLayerInput, cfg.n_layers);
    defer allocator.free(retained_layers);

    for (0..cfg.n_layers) |layer| {
        retained_layers[layer] = .{
            .a = traces[layer].a,
            .scale_a = 1.0, // toy model: unit scale
            .x_attn_i8 = null,
            .scale_x_attn = null,
        };
    }

    const leaf_hash = merkle.hashRetainedStateDirect(retained_layers);
    const leaves = [_]merkle.Hash{leaf_hash};

    var tree = try merkle.buildTree(allocator, &leaves);
    defer tree.deinit();

    // Prove and verify the single token
    var proof = try merkle.prove(allocator, &tree, 0);
    defer proof.deinit();

    if (!merkle.verify(&tree.root, &leaf_hash, &proof)) {
        merkle_proofs_valid = false;
    }

    return .{
        .all_freivalds_passed = all_freivalds_passed,
        .merkle_proofs_valid = merkle_proofs_valid,
        .n_layers_verified = cfg.n_layers,
        .n_matrices_verified = n_matrices_verified,
    };
}

// ===========================================================================
// Tests
// ===========================================================================

test "e2e_pipeline_passes" {
    const allocator = std.testing.allocator;
    const result = try runE2E(allocator);

    try std.testing.expect(result.all_freivalds_passed);
    try std.testing.expect(result.merkle_proofs_valid);
    try std.testing.expectEqual(@as(usize, 2), result.n_layers_verified);
    try std.testing.expectEqual(@as(usize, 14), result.n_matrices_verified); // 7 matrices * 2 layers
}

test "e2e_tampered_weight_detected" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try toy_model.generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try toy_model.generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    // Tamper with a weight AFTER key generation
    model.layers[0].wq[0] = if (model.layers[0].wq[0] == 0) 1 else 0;

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    var traces = try toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (traces) |*t| t.deinit();
        allocator.free(traces);
    }

    // At least one Freivalds check should fail (Wq on layer 0)
    const lt = &traces[0];
    const passed = freivalds.check(
        key.v_vectors[0][0], // v for Wq, layer 0
        lt.x_attn,
        key.rFor(.wq),
        lt.q,
    );
    try std.testing.expect(!passed);
}

test "e2e_io_chain_hashing" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    _ = cfg;

    // Test IO chain: io_0 = H(prompt_hash), io_t = H("vi-io-v4" || leaf_t || token_id_t || io_{t-1})
    const prompt_hash = merkle.hashLeaf("test prompt");
    const leaf0 = merkle.hashLeaf("token0");
    const leaf1 = merkle.hashLeaf("token1");

    const io0 = merkle.ioHashV4(leaf0, 42, prompt_hash);
    const io1 = merkle.ioHashV4(leaf1, 43, io0);

    // IO chain should be deterministic
    const io0_again = merkle.ioHashV4(leaf0, 42, prompt_hash);
    try std.testing.expectEqualSlices(u8, &io0, &io0_again);

    // Different inputs should produce different hashes
    try std.testing.expect(!std.mem.eql(u8, &io0, &io1));

    _ = allocator;
}
