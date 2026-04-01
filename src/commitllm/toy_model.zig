//! Toy model for end-to-end testing of the verification pipeline.
//!
//! Generates random INT8 weights, computes a forward pass with full
//! intermediate capture, generates a verifier key, and verifies it.
//! Validates the entire math pipeline before touching real models.

const std = @import("std");
const field = @import("field.zig");
const constants = @import("constants.zig");
const freivalds = @import("freivalds.zig");
const silu_mod = @import("silu.zig");
const requantize_mod = @import("requantize.zig");
const merkle = @import("merkle.zig");
const types = @import("types.zig");

const Fp = field.Fp;
const ModelConfig = constants.ModelConfig;
const MatrixType = constants.MatrixType;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// All 7 weight matrices for one transformer layer.
pub const LayerWeights = struct {
    wq: []i8,
    wk: []i8,
    wv: []i8,
    wo: []i8,
    wg: []i8,
    wu: []i8,
    wd: []i8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *LayerWeights) void {
        self.allocator.free(self.wq);
        self.allocator.free(self.wk);
        self.allocator.free(self.wv);
        self.allocator.free(self.wo);
        self.allocator.free(self.wg);
        self.allocator.free(self.wu);
        self.allocator.free(self.wd);
        self.* = undefined;
    }

    pub fn getWeight(self: *const LayerWeights, mt: MatrixType) []const i8 {
        return switch (mt) {
            .wq => self.wq,
            .wk => self.wk,
            .wv => self.wv,
            .wo => self.wo,
            .wg => self.wg,
            .wu => self.wu,
            .wd => self.wd,
            .lm_head => unreachable,
        };
    }
};

/// Model with per-layer weights plus an unembedding head.
pub const ToyModel = struct {
    layers: []LayerWeights,
    lm_head: []i8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ToyModel) void {
        for (self.layers) |*lw| {
            lw.deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.lm_head);
        self.* = undefined;
    }
};

const Xoshiro256 = std.Random.Xoshiro256;

fn randomWeights(allocator: std.mem.Allocator, random: std.Random, size: usize) ![]i8 {
    const buf = try allocator.alloc(i8, size);
    for (buf) |*b| {
        b.* = @bitCast(random.int(u8));
    }
    return buf;
}

/// Generate a complete toy model with deterministic weights.
pub fn generateModel(allocator: std.mem.Allocator, cfg: ModelConfig, seed: u64) !ToyModel {
    var prng = Xoshiro256.init(seed);
    const random = prng.random();

    const layers = try allocator.alloc(LayerWeights, cfg.n_layers);
    var initialized: usize = 0;
    errdefer {
        for (layers[0..initialized]) |*lw| lw.deinit();
        allocator.free(layers);
    }

    for (layers) |*lw| {
        lw.allocator = allocator;
        lw.wq = try randomWeights(allocator, random, cfg.hidden_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wq);
        lw.wk = try randomWeights(allocator, random, cfg.kv_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wk);
        lw.wv = try randomWeights(allocator, random, cfg.kv_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wv);
        lw.wo = try randomWeights(allocator, random, cfg.hidden_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wo);
        lw.wg = try randomWeights(allocator, random, cfg.ffn_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wg);
        lw.wu = try randomWeights(allocator, random, cfg.ffn_dim * cfg.hidden_dim);
        errdefer allocator.free(lw.wu);
        lw.wd = try randomWeights(allocator, random, cfg.hidden_dim * cfg.ffn_dim);
        // No errdefer needed for last field — the loop-level errdefer
        // uses `initialized` which hasn't been incremented yet, so this
        // layer won't be double-freed.
        initialized += 1;
    }

    const lm_head = try randomWeights(allocator, random, cfg.vocab_size * cfg.hidden_dim);

    return .{
        .layers = layers,
        .lm_head = lm_head,
        .allocator = allocator,
    };
}

/// INT8 matrix-vector multiply (row-major W, returns i32 accumulators).
///
/// In real inference: INT8 input → matmul (i32 accumulator) → requant (INT8).
/// Freivalds checks the matmul step against the full i32 result.
pub fn matmulI32(allocator: std.mem.Allocator, w: []const i8, x: []const i8, rows: usize, cols: usize) ![]i32 {
    std.debug.assert(w.len == rows * cols);
    std.debug.assert(x.len == cols);
    const result = try allocator.alloc(i32, rows);
    for (0..rows) |r| {
        var acc: i32 = 0;
        for (0..cols) |c| {
            acc += @as(i32, w[r * cols + c]) * @as(i32, x[c]);
        }
        result[r] = acc;
    }
    return result;
}

/// Compute logit vector from last hidden state via lm_head matmul.
pub fn computeLogits(allocator: std.mem.Allocator, lm_head: []const i8, last_hidden: []const i8, vocab_size: usize, hidden_dim: usize) ![]f32 {
    const acc = try matmulI32(allocator, lm_head, last_hidden, vocab_size, hidden_dim);
    defer allocator.free(acc);
    const logits = try allocator.alloc(f32, vocab_size);
    for (acc, 0..) |v, i| {
        logits[i] = @floatFromInt(v);
    }
    return logits;
}

/// Run a single-token forward pass through all layers.
///
/// For a single token (no KV cache), softmax of one score is always 1.0,
/// so the attention output per query head is the V vector of its KV head.
pub fn forwardPass(allocator: std.mem.Allocator, cfg: ModelConfig, model: *const ToyModel, input: []const i8) ![]types.LayerTrace {
    std.debug.assert(input.len == cfg.hidden_dim);

    var x = try allocator.dupe(i8, input);
    errdefer allocator.free(x);
    const heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    const layer_traces = try allocator.alloc(types.LayerTrace, cfg.n_layers);
    var traces_initialized: usize = 0;
    errdefer {
        for (layer_traces[0..traces_initialized]) |*t| t.deinit();
        allocator.free(layer_traces);
    }

    for (model.layers, 0..) |*lw, layer_idx| {
        const x_attn = x;

        const q = try matmulI32(allocator, lw.wq, x_attn, cfg.hidden_dim, cfg.hidden_dim);
        errdefer allocator.free(q);
        const k = try matmulI32(allocator, lw.wk, x_attn, cfg.kv_dim, cfg.hidden_dim);
        errdefer allocator.free(k);
        const v = try matmulI32(allocator, lw.wv, x_attn, cfg.kv_dim, cfg.hidden_dim);
        errdefer allocator.free(v);

        // Single-token GQA: softmax([score]) = [1.0], attn output = V.
        const v_i8 = try requantize_mod.requantizeSlice(allocator, v);
        defer allocator.free(v_i8);

        const a = try allocator.alloc(i8, cfg.hidden_dim);
        errdefer allocator.free(a);
        for (0..cfg.n_q_heads) |qh| {
            const kv_head = qh / heads_per_kv;
            const src_start = kv_head * cfg.d_head;
            const dst_start = qh * cfg.d_head;
            @memcpy(a[dst_start .. dst_start + cfg.d_head], v_i8[src_start .. src_start + cfg.d_head]);
        }

        const attn_out = try matmulI32(allocator, lw.wo, a, cfg.hidden_dim, cfg.hidden_dim);
        errdefer allocator.free(attn_out);

        const x_ffn = try requantize_mod.requantizeSlice(allocator, attn_out);
        errdefer allocator.free(x_ffn);

        const g = try matmulI32(allocator, lw.wg, x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        errdefer allocator.free(g);
        const u = try matmulI32(allocator, lw.wu, x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        errdefer allocator.free(u);
        const h = try silu_mod.computeHUnitScale(allocator, g, u);
        errdefer allocator.free(h);
        const ffn_out = try matmulI32(allocator, lw.wd, h, cfg.hidden_dim, cfg.ffn_dim);
        errdefer allocator.free(ffn_out);

        // Next layer input.
        x = try requantize_mod.requantizeSlice(allocator, ffn_out);

        layer_traces[layer_idx] = .{
            .x_attn = x_attn,
            .q = q,
            .k = k,
            .v = v,
            .a = a,
            .attn_out = attn_out,
            .x_ffn = x_ffn,
            .g = g,
            .u = u,
            .h = h,
            .ffn_out = ffn_out,
            .allocator = allocator,
        };
        traces_initialized += 1;
    }

    // Free the final x (output of last layer, not stored in any trace).
    allocator.free(x);

    return layer_traces;
}

/// Generate a verifier key from model weights.
///
/// This is a VERIFIER-SIDE operation. The returned key contains secret
/// random vectors r_j that must never be shared with the prover.
pub fn generateKey(allocator: std.mem.Allocator, cfg: ModelConfig, model: *const ToyModel, seed: u64) !types.VerifierKey {
    var prng = Xoshiro256.init(seed);
    const random = prng.random();

    // Generate per-matrix-type r vectors (all 8, including LmHead).
    const r_vectors = try allocator.alloc([]Fp, MatrixType.ALL.len);
    var r_initialized: usize = 0;
    errdefer {
        for (r_vectors[0..r_initialized]) |r| allocator.free(r);
        allocator.free(r_vectors);
    }

    for (MatrixType.ALL, 0..) |mt, i| {
        const dim = mt.outputDim(cfg);
        const r = try allocator.alloc(Fp, dim);
        for (r) |*elem| {
            elem.* = Fp.new(random.int(u32));
        }
        r_vectors[i] = r;
        r_initialized += 1;
    }

    // Precompute v = r^T W per layer per matrix type.
    const v_vectors = try allocator.alloc([][]Fp, cfg.n_layers);
    var v_layers_initialized: usize = 0;
    errdefer {
        for (v_vectors[0..v_layers_initialized]) |layer_vs| {
            for (layer_vs) |v| allocator.free(v);
            allocator.free(layer_vs);
        }
        allocator.free(v_vectors);
    }

    for (model.layers) |*lw| {
        const layer_vs = try allocator.alloc([]Fp, MatrixType.PER_LAYER.len);
        var vs_initialized: usize = 0;
        errdefer {
            for (layer_vs[0..vs_initialized]) |v| allocator.free(v);
            allocator.free(layer_vs);
        }

        for (MatrixType.PER_LAYER, 0..) |mt, j| {
            const r = r_vectors[j];
            const rows = mt.outputDim(cfg);
            const cols = mt.inputDim(cfg);
            const w = lw.getWeight(mt);
            layer_vs[j] = try freivalds.precomputeV(allocator, r, w, rows, cols);
            vs_initialized += 1;
        }
        v_vectors[v_layers_initialized] = layer_vs;
        v_layers_initialized += 1;
    }

    // Compute weight-chain hash with domain separator.
    _ = computeWeightHash(model);

    // Expand 64-bit seed to 32 bytes via SHA-256 for the key seed.
    var seed_bytes: [32]u8 = undefined;
    var hasher = Sha256.init(.{});
    hasher.update("vi-seed-expand-v1");
    hasher.update(&std.mem.toBytes(seed));
    seed_bytes = hasher.finalResult();

    return .{
        .config = cfg,
        .seed = seed_bytes,
        .r_vectors = r_vectors,
        .v_vectors = v_vectors,
        .allocator = allocator,
    };
}

/// Generate a Level B verifier key with W_o norms and optional lm_head.
pub fn generateKeyLevelB(
    allocator: std.mem.Allocator,
    cfg: ModelConfig,
    model: *const ToyModel,
    seed: u64,
    include_lm_head: bool,
) !types.VerifierKey {
    var key = try generateKey(allocator, cfg, model, seed);

    if (include_lm_head) {
        const r = key.rFor(.lm_head);
        key.v_lm_head = try freivalds.precomputeV(allocator, r, model.lm_head, cfg.vocab_size, cfg.hidden_dim);
        key.lm_head = try allocator.dupe(i8, model.lm_head);
    }

    return key;
}

fn computeWeightHash(model: *const ToyModel) merkle.Hash {
    var hasher = Sha256.init(.{});
    hasher.update("vi-weight-chain-v1");
    hasher.update("I8");
    for (model.layers) |*lw| {
        for (MatrixType.PER_LAYER) |mt| {
            const w = lw.getWeight(mt);
            hasher.update(std.mem.sliceAsBytes(w));
        }
    }
    return hasher.finalResult();
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test "matmul_i32_basic" {
    const allocator = std.testing.allocator;
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const x = [_]i8{ 7, 8, 9 };
    const result = try matmulI32(allocator, &w, &x, 2, 3);
    defer allocator.free(result);
    try std.testing.expectEqual(@as(i32, 50), result[0]);
    try std.testing.expectEqual(@as(i32, 122), result[1]);
}

test "generate_model" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();
    try std.testing.expectEqual(@as(usize, 2), model.layers.len);
    try std.testing.expectEqual(cfg.hidden_dim * cfg.hidden_dim, model.layers[0].wq.len);
    try std.testing.expectEqual(cfg.vocab_size * cfg.hidden_dim, model.lm_head.len);
}

test "generate_model_deterministic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var m1 = try generateModel(allocator, cfg, 42);
    defer m1.deinit();
    var m2 = try generateModel(allocator, cfg, 42);
    defer m2.deinit();
    try std.testing.expectEqualSlices(i8, m1.layers[0].wq, m2.layers[0].wq);
    try std.testing.expectEqualSlices(i8, m1.lm_head, m2.lm_head);
}

test "forward_pass_basic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }

    try std.testing.expectEqual(@as(usize, 2), traces.len);
    try std.testing.expectEqual(cfg.hidden_dim, traces[0].x_attn.len);
    try std.testing.expectEqual(cfg.hidden_dim, traces[0].q.len);
    try std.testing.expectEqual(cfg.ffn_dim, traces[0].g.len);
}

test "generate_key_and_verify_structure" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    try std.testing.expectEqual(@as(usize, 8), key.r_vectors.len);
    try std.testing.expectEqual(@as(usize, 2), key.v_vectors.len);
    try std.testing.expectEqual(@as(usize, 7), key.v_vectors[0].len);
}

test "freivalds_e2e_with_toy_model" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }

    // Verify all 7 × 2 = 14 Freivalds checks.
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
            try std.testing.expect(freivalds.check(
                layer_vs[chk.v_idx],
                chk.input,
                key.rFor(chk.r_mt),
                chk.output,
            ));
        }
    }
}

test "weight_tamper_detected" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    var key = try generateKey(allocator, cfg, &model, 42);
    defer key.deinit();

    // Tamper AFTER keygen.
    model.layers[0].wq[0] = if (model.layers[0].wq[0] == 0) 1 else 0;

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }

    // Wq check on layer 0 should FAIL.
    const passed = freivalds.check(
        key.v_vectors[0][0],
        traces[0].x_attn,
        key.rFor(.wq),
        traces[0].q,
    );
    try std.testing.expect(!passed);
}

test "compute_logits_basic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const logits = try computeLogits(allocator, model.lm_head, &input, cfg.vocab_size, cfg.hidden_dim);
    defer allocator.free(logits);
    try std.testing.expectEqual(cfg.vocab_size, logits.len);
}

test "weight_hash_deterministic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var m1 = try generateModel(allocator, cfg, 42);
    defer m1.deinit();
    var m2 = try generateModel(allocator, cfg, 42);
    defer m2.deinit();

    const h1 = computeWeightHash(&m1);
    const h2 = computeWeightHash(&m2);
    try std.testing.expectEqualSlices(u8, &h1, &h2);
}
