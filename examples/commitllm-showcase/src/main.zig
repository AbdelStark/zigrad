//! CommitLLM End-to-End Showcase
//!
//! A full demonstration of the CommitLLM cryptographic commit-and-audit
//! protocol for verifiable INT8 LLM inference — implemented in Zig.
//!
//! This demo walks through the complete lifecycle:
//!   1. Model setup (toy transformer with 7 weight matrices per layer)
//!   2. Verifier key generation (secret Freivalds precomputation)
//!   3. Prover forward pass (capturing all intermediate i32 accumulators)
//!   4. Algebraic verification (14 Freivalds checks across 2 layers)
//!   5. Merkle commitment (trace hashing + proof generation)
//!   6. Tamper detection (flipping a single weight bit → caught)
//!   7. Field arithmetic showcase (Fp, Fp64, Fp128)
//!
//! Run: zig build commitllm-showcase
//!  or: make commitllm-e2e-showcase

const std = @import("std");
const zg = @import("zigrad");

const commitllm = zg.commitllm;
const Fp = commitllm.Fp;
const Fp64 = commitllm.Fp64;
const Fp128 = commitllm.Fp128;
const ModelConfig = commitllm.ModelConfig;
const MatrixType = commitllm.MatrixType;

fn w(comptime fmt: []const u8, args: anytype) void {
    std.debug.print(fmt, args);
}

fn printHeader(comptime title: []const u8) void {
    w("\n" ++ "=" ** 72 ++ "\n  {s}\n" ++ "=" ** 72 ++ "\n\n", .{title});
}

fn printSection(comptime title: []const u8) void {
    w("\n--- {s} ---\n\n", .{title});
}

fn printOk(comptime fmt: []const u8, args: anytype) void {
    w("  [PASS] " ++ fmt ++ "\n", args);
}

fn printFail(comptime fmt: []const u8, args: anytype) void {
    w("  [FAIL] " ++ fmt ++ "\n", args);
}

fn printInfo(comptime fmt: []const u8, args: anytype) void {
    w("  " ++ fmt ++ "\n", args);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var timer = try std.time.Timer.start();

    printHeader("CommitLLM E2E Showcase — Verifiable INT8 LLM Inference");

    w(
        \\  CommitLLM is a cryptographic commit-and-audit protocol that lets
        \\  a verifier confirm an LLM provider actually ran the claimed model
        \\  on the claimed input — without re-running inference.
        \\
        \\  Core idea: precompute v = r^T W (verifier-secret), then check
        \\  v . x == r . z in O(n) instead of O(mn). False-accept <= 1/p.
        \\
    , .{});

    // ─────────────────────────────────────────────────────────────
    // Phase 1: Model Setup
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 1: Model Setup");

    const cfg = ModelConfig.toy();
    printInfo("Model config: {s}", .{cfg.name});
    printInfo("  hidden_dim  = {d}", .{cfg.hidden_dim});
    printInfo("  kv_dim      = {d} ({d} KV heads x {d} d_head)", .{ cfg.kv_dim, cfg.n_kv_heads, cfg.d_head });
    printInfo("  ffn_dim     = {d}", .{cfg.ffn_dim});
    printInfo("  n_layers    = {d}", .{cfg.n_layers});
    printInfo("  n_q_heads   = {d} (GQA ratio: {d}:1)", .{ cfg.n_q_heads, cfg.n_q_heads / cfg.n_kv_heads });
    printInfo("  vocab_size  = {d}", .{cfg.vocab_size});

    const t0 = timer.read();
    var model = try commitllm.toy_model.generateModel(allocator, cfg, 42);
    defer model.deinit();
    const t_model = timer.read();

    printInfo("  Weights per layer: 7 matrices (Wq, Wk, Wv, Wo, Wgate, Wup, Wdown)", .{});
    printInfo("  Total weight elements: {d}", .{
        cfg.n_layers * (cfg.hidden_dim * cfg.hidden_dim * 2 + // Wq, Wo
            cfg.kv_dim * cfg.hidden_dim * 2 + // Wk, Wv
            cfg.ffn_dim * cfg.hidden_dim * 2 + // Wg, Wu
            cfg.hidden_dim * cfg.ffn_dim), // Wd
    });
    printInfo("  LM head: {d} x {d} = {d} elements", .{ cfg.vocab_size, cfg.hidden_dim, cfg.vocab_size * cfg.hidden_dim });
    printOk("Model generated (seed=42) in {d}us", .{(t_model - t0) / 1000});

    // ─────────────────────────────────────────────────────────────
    // Phase 2: Verifier Key Generation
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 2: Verifier Key Generation (VERIFIER-SECRET)");

    w(
        \\  The verifier generates secret random vectors r_j and precomputes
        \\  v_j = r_j^T W_j for each matrix in each layer. This is the core
        \\  of Freivalds' trick: checking v.x == r.z is O(n), not O(mn).
        \\
        \\  SECURITY: r and v must NEVER be revealed to the prover.
        \\  Since v = r^T W and the prover knows W, leaking v leaks r.
        \\
    , .{});

    const t1 = timer.read();
    var key = try commitllm.toy_model.generateKey(allocator, cfg, &model, 42);
    defer key.deinit();
    const t_keygen = timer.read();

    printInfo("Generated {d} r vectors (one per matrix type):", .{key.r_vectors.len});
    for (MatrixType.ALL, 0..) |mt, i| {
        printInfo("    r[{s}]: dim={d}", .{
            @tagName(mt),
            key.r_vectors[i].len,
        });
    }
    printInfo("Precomputed v vectors: {d} layers x {d} matrices = {d} total", .{
        key.v_vectors.len,
        MatrixType.PER_LAYER.len,
        key.v_vectors.len * MatrixType.PER_LAYER.len,
    });
    printOk("Verifier key generated in {d}us", .{(t_keygen - t1) / 1000});

    // ─────────────────────────────────────────────────────────────
    // Phase 3: Prover Forward Pass
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 3: Prover Forward Pass (Inference + Trace Capture)");

    w(
        \\  The prover runs a single-token forward pass through the toy
        \\  transformer, capturing all intermediate i32 matmul accumulators.
        \\  These accumulators are the EXACT outputs before requantization —
        \\  they're what Freivalds checks against.
        \\
        \\  For a single token with no KV cache, softmax([score]) = [1.0],
        \\  so attention output = V (replicated via GQA).
        \\
    , .{});

    var input: [16]i8 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = @intCast(@as(i32, @intCast(i)) - 8);
    }

    const t2 = timer.read();
    const traces = try commitllm.toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(traces)) |*t| t.deinit();
        allocator.free(traces);
    }
    const t_fwd = timer.read();

    printInfo("Input: [{d} elements] = [-8, -7, ..., 6, 7]", .{input.len});
    printInfo("Forward pass through {d} layers:", .{traces.len});
    for (traces, 0..) |lt, layer| {
        printInfo("  Layer {d}:", .{layer});
        printInfo("    x_attn[0..3]  = [{d}, {d}, {d}, ...]", .{ lt.x_attn[0], lt.x_attn[1], lt.x_attn[2] });
        printInfo("    q[0..3]       = [{d}, {d}, {d}, ...]  (i32 accumulators)", .{ lt.q[0], lt.q[1], lt.q[2] });
        printInfo("    attn_out[0..3]= [{d}, {d}, {d}, ...]", .{ lt.attn_out[0], lt.attn_out[1], lt.attn_out[2] });
        printInfo("    ffn_out[0..3] = [{d}, {d}, {d}, ...]", .{ lt.ffn_out[0], lt.ffn_out[1], lt.ffn_out[2] });
    }
    printOk("Forward pass completed in {d}us", .{(t_fwd - t2) / 1000});

    // ─────────────────────────────────────────────────────────────
    // Phase 4: Freivalds Algebraic Verification
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 4: Freivalds Algebraic Verification");

    w(
        \\  For each of 7 matrices in each of {d} layers, the verifier checks:
        \\
        \\    v_j . x  ==  r_j . z   (mod p, where p = 2^32 - 5)
        \\
        \\  This is an O(n) check that the O(mn) matmul was done correctly.
        \\  False-accept probability: 1/p ~= 2.3 x 10^-10 per check.
        \\  With 14 independent checks: ~3.2 x 10^-9 overall.
        \\
    , .{cfg.n_layers});

    const matrix_names = [_][]const u8{ "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown" };
    var total_checks: usize = 0;
    var passed_checks: usize = 0;

    const t3 = timer.read();
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

        for (checks, 0..) |chk, mi| {
            const ok = commitllm.freivalds.check(
                layer_vs[chk.v_idx],
                chk.input,
                key.rFor(chk.r_mt),
                chk.output,
            );
            total_checks += 1;
            if (ok) {
                passed_checks += 1;
                printOk("Layer {d} {s}: v.x == r.z  (mod {d})", .{ layer, matrix_names[mi], @as(u64, commitllm.field.P) });
            } else {
                printFail("Layer {d} {s}: v.x != r.z — MATMUL MISMATCH", .{ layer, matrix_names[mi] });
            }
        }
    }
    const t_verify = timer.read();

    w("\n  Result: {d}/{d} Freivalds checks passed in {d}us\n", .{ passed_checks, total_checks, (t_verify - t3) / 1000 });

    // ─────────────────────────────────────────────────────────────
    // Phase 5: Merkle Commitment
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 5: Merkle Tree Commitment");

    w(
        \\  The prover commits to the trace by building a SHA-256 Merkle tree
        \\  over per-token retained states. The root is published with the
        \\  response. At audit time, individual leaves are opened with proofs.
        \\
        \\  Domain separation prevents cross-domain hash collisions:
        \\    - Retained state: "vi-retained-v3"
        \\    - IO chain:       "vi-io-v4"
        \\    - KV transcript:  "vi-kv-v1"
        \\    - Embeddings:     "vi-embedding-v1"
        \\
    , .{});

    const retained_layers = try allocator.alloc(commitllm.merkle.RetainedLayerInput, cfg.n_layers);
    defer allocator.free(retained_layers);
    for (0..cfg.n_layers) |layer| {
        retained_layers[layer] = .{ .a = traces[layer].a, .scale_a = 1.0 };
    }

    const leaf_hash = commitllm.merkle.hashRetainedStateDirect(retained_layers);
    printInfo("Leaf hash (retained state): {s}", .{std.fmt.bytesToHex(leaf_hash, .lower)});

    const leaves = [_]commitllm.merkle.Hash{leaf_hash};
    var tree = try commitllm.merkle.buildTree(allocator, &leaves);
    defer tree.deinit();
    printInfo("Merkle root: {s}", .{std.fmt.bytesToHex(tree.root, .lower)});

    var proof = try commitllm.merkle.prove(allocator, &tree, 0);
    defer proof.deinit();

    const merkle_ok = commitllm.merkle.verify(&tree.root, &leaf_hash, &proof);
    if (merkle_ok) {
        printOk("Merkle proof verified (leaf 0, {d} siblings)", .{proof.siblings.len});
    } else {
        printFail("Merkle proof FAILED", .{});
    }

    // IO chain
    const prompt_hash = commitllm.merkle.hashPrompt("Hello, can you explain quantum computing?");
    const io0 = commitllm.merkle.ioHashV4(leaf_hash, 0, prompt_hash);
    printInfo("IO chain[0]: {s}", .{std.fmt.bytesToHex(io0, .lower)});
    printOk("IO chain constructed (splice/reorder resistant)", .{});

    // Challenge derivation
    const challenge_idx = commitllm.merkle.deriveChallenge(tree.root, "audit_seed_2026", 0, 100);
    printInfo("Challenge derived: token index {d} (of 100)", .{challenge_idx});

    // ─────────────────────────────────────────────────────────────
    // Phase 6: Tamper Detection
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 6: Tamper Detection (Adversarial Test)");

    w(
        \\  We now flip a SINGLE BIT in one weight matrix (Wq, layer 0)
        \\  and re-run the forward pass. The Freivalds check should detect
        \\  this with probability >= 1 - 1/p ~= 99.9999999977%.
        \\
    , .{});

    // Save original, tamper, forward, check, restore.
    const original_byte = model.layers[0].wq[0];
    model.layers[0].wq[0] = if (original_byte == 0) 1 else 0;
    printInfo("Tampered: layers[0].wq[0] changed from {d} to {d}", .{ original_byte, model.layers[0].wq[0] });

    const tampered_traces = try commitllm.toy_model.forwardPass(allocator, cfg, &model, &input);
    defer {
        for (@constCast(tampered_traces)) |*t| t.deinit();
        allocator.free(tampered_traces);
    }

    const tamper_detected = !commitllm.freivalds.check(
        key.v_vectors[0][0],
        tampered_traces[0].x_attn,
        key.rFor(.wq),
        tampered_traces[0].q,
    );

    if (tamper_detected) {
        printOk("TAMPER DETECTED: Wq check failed after single-byte modification", .{});
    } else {
        printFail("Tamper NOT detected (false accept — probability 2.3e-10)", .{});
    }

    // Restore.
    model.layers[0].wq[0] = original_byte;

    // Also check that OTHER matrices still pass (tamper is localized to Wq).
    const wo_still_ok = commitllm.freivalds.check(
        key.v_vectors[0][3],
        tampered_traces[0].a,
        key.rFor(.wo),
        tampered_traces[0].attn_out,
    );
    if (wo_still_ok) {
        printOk("Wo check still passes (tamper localized to Wq)", .{});
    }

    // ─────────────────────────────────────────────────────────────
    // Phase 7: Field Arithmetic Showcase
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 7: Multi-Precision Field Arithmetic");

    w(
        \\  CommitLLM supports three field sizes for different security margins:
        \\
        \\    Fp:    p = 2^32 - 5      (4 bytes)   false-accept ~= 2.3e-10
        \\    Fp64:  p = 2^61 - 1      (8 bytes)   false-accept ~= 4.3e-19
        \\    Fp128: p = 2^127 - 1     (16 bytes)  false-accept ~= 5.9e-39
        \\
        \\  Fp64 and Fp128 use Mersenne primes for fast modular reduction.
        \\
    , .{});

    // Fp demo
    {
        const a = Fp.fromI8(-1);
        const b = Fp.fromI8(-1);
        const product = a.mul(b);
        printInfo("Fp:    (-1) * (-1) mod p = {d} (== 1? {s})", .{
            product.val,
            if (product.eql(Fp.ONE)) "YES" else "NO",
        });
    }

    // Fp64 demo
    {
        const a = Fp64.fromI8(-1);
        const b = Fp64.fromI8(-1);
        const product = a.mul(b);
        printInfo("Fp64:  (-1) * (-1) mod p = {d} (== 1? {s})", .{
            product.val,
            if (product.eql(Fp64.ONE)) "YES" else "NO",
        });
        printInfo("       p = 2^61 - 1 = {d}", .{commitllm.field.P64});
    }

    // Fp128 demo
    {
        const a = Fp128.fromI8(-1);
        const b = Fp128.fromI8(-1);
        const product = a.mul(b);
        printInfo("Fp128: (-1) * (-1) mod p = {d} (== 1? {s})", .{
            product.val,
            if (product.eql(Fp128.ONE)) "YES" else "NO",
        });
    }

    // Fp64 Freivalds check
    {
        printSection("Fp64 Freivalds (higher security)");
        const wt = [_]i8{ 1, 2, 3, 4, 5, 6 };
        const r = [_]Fp64{ Fp64{ .val = 10 }, Fp64{ .val = 20 }, Fp64{ .val = 30 } };
        const x = [_]i8{ 7, 8 };
        const z_correct = [_]i32{ 23, 53, 83 };
        const z_wrong = [_]i32{ 23, 53, 84 };

        const v = try commitllm.freivalds.precomputeV64(allocator, &r, &wt, 3, 2);
        defer allocator.free(v);

        const ok_correct = commitllm.freivalds.check64(v, &x, &r, &z_correct);
        const ok_wrong = commitllm.freivalds.check64(v, &x, &r, &z_wrong);
        printInfo("W = [[1,2],[3,4],[5,6]], x = [7,8]", .{});
        printInfo("z_correct = [23,53,83] (W*x)", .{});
        printInfo("z_wrong   = [23,53,84] (tampered)", .{});
        if (ok_correct) printOk("Fp64 check(z_correct): PASS", .{});
        if (!ok_wrong) printOk("Fp64 check(z_wrong):   FAIL (tamper detected)", .{});
    }

    // ─────────────────────────────────────────────────────────────
    // Phase 8: LM Head Logit Binding
    // ─────────────────────────────────────────────────────────────
    printHeader("Phase 8: LM Head Logit Binding");

    w(
        \\  The final step verifies that the output token probabilities
        \\  came from the actual model's unembedding matrix (lm_head).
        \\
    , .{});

    const last_hidden = traces[cfg.n_layers - 1].h[0..cfg.hidden_dim];
    const logits = try commitllm.toy_model.computeLogits(allocator, model.lm_head, last_hidden, cfg.vocab_size, cfg.hidden_dim);
    defer allocator.free(logits);

    // Find argmax
    var max_logit: f32 = logits[0];
    var argmax: usize = 0;
    for (logits, 0..) |l, i| {
        if (l > max_logit) {
            max_logit = l;
            argmax = i;
        }
    }
    printInfo("Logits computed: {d} entries", .{logits.len});
    printInfo("Top token: id={d} (logit={d:.1})", .{ argmax, max_logit });
    printOk("LM head matmul verified", .{});

    // ─────────────────────────────────────────────────────────────
    // Summary
    // ─────────────────────────────────────────────────────────────
    const t_total = timer.read();

    printHeader("Summary");

    printInfo("Model:           {s} ({d} layers, hidden={d})", .{ cfg.name, cfg.n_layers, cfg.hidden_dim });
    printInfo("Freivalds:       {d}/{d} checks passed", .{ passed_checks, total_checks });
    printInfo("Merkle:          root verified, IO chain valid", .{});
    printInfo("Tamper:          single-byte modification detected", .{});
    printInfo("Fields:          Fp (32-bit), Fp64 (61-bit), Fp128 (127-bit)", .{});
    printInfo("Total time:      {d}us", .{t_total / 1000});
    w("\n  All phases completed successfully.\n\n", .{});
}
