//! Toy model for end-to-end testing of the verification pipeline.
//!
//! Generates random INT8 weights, computes a fake forward pass,
//! generates a verifier key, produces a trace, and verifies it.
//! This validates the entire math pipeline before touching real models.

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

/// All 7 weight matrices for one layer.
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

    /// Get weight matrix for a given per-layer matrix type.
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

/// Model with per-layer weights plus an unembedding head (lm_head).
pub const ToyModel = struct {
    layers: []LayerWeights,
    lm_head: []i8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ToyModel) void {
        for (self.layers) |*lw| {
            @constCast(lw).deinit();
        }
        self.allocator.free(self.layers);
        self.allocator.free(self.lm_head);
        self.* = undefined;
    }
};

/// Simple xoshiro256** PRNG for deterministic weight generation.
/// We use this instead of ChaCha20 because we need to match our own
/// test vectors, not the Rust ChaCha20 stream (which differs in buffering).
const Xoshiro256 = std.Random.Xoshiro256;

/// Generate random INT8 weights of given size.
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
    errdefer {
        for (layers[0..]) |*lw| lw.deinit();
        allocator.free(layers);
    }

    for (layers, 0..) |*lw, i| {
        _ = i;
        lw.* = .{
            .wq = try randomWeights(allocator, random, cfg.hidden_dim * cfg.hidden_dim),
            .wk = try randomWeights(allocator, random, cfg.kv_dim * cfg.hidden_dim),
            .wv = try randomWeights(allocator, random, cfg.kv_dim * cfg.hidden_dim),
            .wo = try randomWeights(allocator, random, cfg.hidden_dim * cfg.hidden_dim),
            .wg = try randomWeights(allocator, random, cfg.ffn_dim * cfg.hidden_dim),
            .wu = try randomWeights(allocator, random, cfg.ffn_dim * cfg.hidden_dim),
            .wd = try randomWeights(allocator, random, cfg.hidden_dim * cfg.ffn_dim),
            .allocator = allocator,
        };
    }

    const lm_head = try randomWeights(allocator, random, cfg.vocab_size * cfg.hidden_dim);

    return .{
        .layers = layers,
        .lm_head = lm_head,
        .allocator = allocator,
    };
}

/// INT8 matrix-vector multiply (row-major W, returns i32 accumulators).
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

/// Requantize an i32 slice to i8 slice. Caller owns returned memory.
fn requantizeSlice(allocator: std.mem.Allocator, acc: []const i32) ![]i8 {
    return requantize_mod.requantizeSlice(allocator, acc);
}

/// Run a single-token forward pass through all layers.
///
/// For a single token (no KV cache), softmax of one score is always 1.0,
/// so the attention output per query head is the V vector of its KV head.
pub fn forwardPass(allocator: std.mem.Allocator, cfg: ModelConfig, model: *const ToyModel, input: []const i8) ![]types.LayerTrace {
    std.debug.assert(input.len == cfg.hidden_dim);

    var x = try allocator.dupe(i8, input);
    const heads_per_kv = cfg.n_q_heads / cfg.n_kv_heads;

    const layer_traces = try allocator.alloc(types.LayerTrace, cfg.n_layers);
    errdefer allocator.free(layer_traces);

    for (model.layers, 0..) |*lw, layer_idx| {
        const x_attn = x;

        // Attention projections
        const q = try matmulI32(allocator, lw.wq, x_attn, cfg.hidden_dim, cfg.hidden_dim);
        const k = try matmulI32(allocator, lw.wk, x_attn, cfg.kv_dim, cfg.hidden_dim);
        const v = try matmulI32(allocator, lw.wv, x_attn, cfg.kv_dim, cfg.hidden_dim);

        // Single-token GQA: softmax([score]) = [1.0], so attn output = V.
        const v_i8 = try requantizeSlice(allocator, v);
        defer allocator.free(v_i8);

        const a = try allocator.alloc(i8, cfg.hidden_dim);
        for (0..cfg.n_q_heads) |qh| {
            const kv_head = qh / heads_per_kv;
            const src_start = kv_head * cfg.d_head;
            const dst_start = qh * cfg.d_head;
            @memcpy(a[dst_start .. dst_start + cfg.d_head], v_i8[src_start .. src_start + cfg.d_head]);
        }

        const attn_out = try matmulI32(allocator, lw.wo, a, cfg.hidden_dim, cfg.hidden_dim);

        // Simplified residual: just requantize
        const x_ffn = try requantizeSlice(allocator, attn_out);

        // FFN
        const g = try matmulI32(allocator, lw.wg, x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        const u = try matmulI32(allocator, lw.wu, x_ffn, cfg.ffn_dim, cfg.hidden_dim);
        const h = try silu_mod.computeHUnitScale(allocator, g, u);
        const ffn_out = try matmulI32(allocator, lw.wd, h, cfg.hidden_dim, cfg.ffn_dim);

        // Next layer input
        x = try requantizeSlice(allocator, ffn_out);

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
    }

    // Free the final x (which is not stored in any trace)
    allocator.free(x);

    return layer_traces;
}

/// Generate a verifier key from model weights.
pub fn generateKey(allocator: std.mem.Allocator, cfg: ModelConfig, model: *const ToyModel, seed: u64) !types.VerifierKey {
    var prng = Xoshiro256.init(seed);
    const random = prng.random();

    // Generate per-matrix-type r vectors (all 8, including LmHead)
    const r_vectors = try allocator.alloc([]Fp, MatrixType.ALL.len);
    errdefer {
        for (r_vectors) |r| allocator.free(r);
        allocator.free(r_vectors);
    }

    for (MatrixType.ALL, 0..) |mt, i| {
        const dim = mt.outputDim(cfg);
        const r = try allocator.alloc(Fp, dim);
        for (r) |*elem| {
            elem.* = Fp.new(random.int(u32));
        }
        r_vectors[i] = r;
    }

    // Precompute v_j^(i) = r_j^T W_j^(i) for each layer and per-layer matrix type
    const v_vectors = try allocator.alloc([][]Fp, cfg.n_layers);
    errdefer {
        for (v_vectors) |layer_vs| {
            for (layer_vs) |v| allocator.free(v);
            allocator.free(layer_vs);
        }
        allocator.free(v_vectors);
    }

    for (model.layers, 0..) |*lw, layer| {
        const layer_vs = try allocator.alloc([]Fp, MatrixType.PER_LAYER.len);
        for (MatrixType.PER_LAYER, 0..) |mt, j| {
            const r = r_vectors[j];
            const rows = mt.outputDim(cfg);
            const cols = mt.inputDim(cfg);
            const w = lw.getWeight(mt);
            layer_vs[j] = try freivalds.precomputeV(allocator, r, w, rows, cols);
        }
        v_vectors[layer] = layer_vs;
    }

    // Compute weight chain hash
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    for (model.layers) |*lw| {
        for (MatrixType.PER_LAYER) |mt| {
            const w = lw.getWeight(mt);
            hasher.update(std.mem.sliceAsBytes(w));
        }
    }
    const weight_hash = hasher.finalResult();

    return .{
        .config = cfg,
        .seed = std.mem.toBytes(seed) ++ std.mem.toBytes(seed) ++ std.mem.toBytes(seed) ++ std.mem.toBytes(seed),
        .r_vectors = r_vectors,
        .v_vectors = v_vectors,
        .v_lm_head = null,
        .weight_hash = weight_hash,
        .allocator = allocator,
    };
}

// ===========================================================================
// Tests
// ===========================================================================

test "matmul_i32_basic" {
    const allocator = std.testing.allocator;
    // 2x3 matrix * 3-vector
    const w = [_]i8{ 1, 2, 3, 4, 5, 6 };
    const x = [_]i8{ 7, 8, 9 };
    const result = try matmulI32(allocator, &w, &x, 2, 3);
    defer allocator.free(result);
    // [1*7+2*8+3*9, 4*7+5*8+6*9] = [50, 122]
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
    try std.testing.expectEqual(cfg.kv_dim * cfg.hidden_dim, model.layers[0].wk.len);
    try std.testing.expectEqual(cfg.vocab_size * cfg.hidden_dim, model.lm_head.len);
}

test "forward_pass_basic" {
    const allocator = std.testing.allocator;
    const cfg = ModelConfig.toy();
    var model = try generateModel(allocator, cfg, 42);
    defer model.deinit();

    // Create a simple input
    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const traces = try forwardPass(allocator, cfg, &model, &input);
    defer {
        for (traces) |*t| t.deinit();
        allocator.free(traces);
    }

    try std.testing.expectEqual(@as(usize, 2), traces.len);
    try std.testing.expectEqual(cfg.hidden_dim, traces[0].x_attn.len);
    try std.testing.expectEqual(cfg.hidden_dim, traces[0].q.len);
    try std.testing.expectEqual(cfg.ffn_dim, traces[0].g.len);
}

test "generate_key_and_verify" {
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
        for (traces) |*t| t.deinit();
        allocator.free(traces);
    }

    // Verify Freivalds checks for each layer, each matrix type
    for (0..cfg.n_layers) |layer| {
        const lt = &traces[layer];
        const layer_vs = key.v_vectors[layer];

        // Wq check: v_wq . x_attn == r_wq . q
        try std.testing.expect(freivalds.check(
            layer_vs[0], // v for Wq
            lt.x_attn,
            key.rFor(.wq),
            lt.q,
        ));

        // Wk check
        try std.testing.expect(freivalds.check(
            layer_vs[1],
            lt.x_attn,
            key.rFor(.wk),
            lt.k,
        ));

        // Wv check
        try std.testing.expect(freivalds.check(
            layer_vs[2],
            lt.x_attn,
            key.rFor(.wv),
            lt.v,
        ));

        // Wo check
        try std.testing.expect(freivalds.check(
            layer_vs[3],
            lt.a,
            key.rFor(.wo),
            lt.attn_out,
        ));

        // Wg check
        try std.testing.expect(freivalds.check(
            layer_vs[4],
            lt.x_ffn,
            key.rFor(.wg),
            lt.g,
        ));

        // Wu check
        try std.testing.expect(freivalds.check(
            layer_vs[5],
            lt.x_ffn,
            key.rFor(.wu),
            lt.u,
        ));

        // Wd check
        try std.testing.expect(freivalds.check(
            layer_vs[6],
            lt.h,
            key.rFor(.wd),
            lt.ffn_out,
        ));
    }
}
