//! End-to-end pipeline: keygen → forward → commit → verify.
//!
//! Integration test proving all building blocks compose correctly.

const std = @import("std");
const field = @import("field.zig");
const constants = @import("constants.zig");
const freivalds = @import("freivalds.zig");
const merkle = @import("merkle.zig");
const types = @import("types.zig");
const toy_model = @import("toy_model.zig");

const ModelConfig = constants.ModelConfig;
const MatrixType = constants.MatrixType;

/// Result of running the e2e pipeline.
pub const PipelineResult = struct {
    all_freivalds_passed: bool,
    merkle_proofs_valid: bool,
    io_chain_valid: bool,
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

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }

    // Phase 1: Freivalds verification — all 7 matrices × n_layers.
    var all_freivalds_passed = true;
    var n_matrices_verified: usize = 0;

    for (0..cfg.n_layers) |layer| {
        const lt = &traces[layer];
        const layer_vs = key.v_vectors[layer];

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
            if (!freivalds.check(layer_vs[chk.v_idx], chk.input, key.rFor(chk.r_mt), chk.output)) {
                all_freivalds_passed = false;
            }
            n_matrices_verified += 1;
        }
    }

    // Phase 2: Merkle tree commitment and proof verification.
    var merkle_proofs_valid = true;

    const retained_layers = try allocator.alloc(merkle.RetainedLayerInput, cfg.n_layers);
    defer allocator.free(retained_layers);

    for (0..cfg.n_layers) |layer| {
        retained_layers[layer] = .{
            .a = traces[layer].a,
            .scale_a = 1.0,
        };
    }

    const leaf_hash = merkle.hashRetainedStateDirect(retained_layers);
    const leaves = [_]merkle.Hash{leaf_hash};

    var tree = try merkle.buildTree(allocator, &leaves);
    defer tree.deinit();

    var proof = try merkle.prove(allocator, &tree, 0);
    defer proof.deinit();

    if (!merkle.verify(&tree.root, &leaf_hash, &proof)) {
        merkle_proofs_valid = false;
    }

    // Phase 3: IO chain.
    const prompt_hash = merkle.hashPrompt("test prompt");
    const io0 = merkle.ioHashV4(leaf_hash, 0, prompt_hash);
    // Verify chain is deterministic.
    const io0_again = merkle.ioHashV4(leaf_hash, 0, prompt_hash);
    const io_chain_valid = std.mem.eql(u8, &io0, &io0_again);

    return .{
        .all_freivalds_passed = all_freivalds_passed,
        .merkle_proofs_valid = merkle_proofs_valid,
        .io_chain_valid = io_chain_valid,
        .n_layers_verified = cfg.n_layers,
        .n_matrices_verified = n_matrices_verified,
    };
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "e2e_pipeline_passes" {
    const allocator = std.testing.allocator;
    const result = try runE2E(allocator);

    try std.testing.expect(result.all_freivalds_passed);
    try std.testing.expect(result.merkle_proofs_valid);
    try std.testing.expect(result.io_chain_valid);
    try std.testing.expectEqual(@as(usize, 2), result.n_layers_verified);
    try std.testing.expectEqual(@as(usize, 14), result.n_matrices_verified);
}

test "e2e_tampered_weight_detected" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try toy_model.generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try toy_model.generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    // Tamper AFTER keygen.
    model.layers[0].wq[0] = if (model.layers[0].wq[0] == 0) 1 else 0;

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }

    // Wq on layer 0 should fail.
    const passed = freivalds.check(
        key.v_vectors[0][0],
        traces[0].x_attn,
        key.rFor(.wq),
        traces[0].q,
    );
    try std.testing.expect(!passed);
}
