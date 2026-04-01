//! CommitLLM Cross-Language Prover (Rust)
//!
//! Generates a complete proof bundle that the Zig verifier can check:
//!   1. Toy model weights (deterministic LCG)
//!   2. Verifier key (r vectors, v = r^T W precomputation)
//!   3. Forward pass trace (all i32 accumulators)
//!   4. Merkle commitment (retained state hashes, tree root, proofs)
//!   5. IO chain (domain-separated sequential hash chain)
//!
//! The Zig verifier loads this JSON and independently verifies every check.
//! If both implementations are correct and compatible, all checks pass.

use serde::Serialize;
use sha2::{Digest, Sha256};

// ═══════════════════════════════════════════════════════════════════════
// Field arithmetic — must be bit-identical to Zig's field.zig
// ═══════════════════════════════════════════════════════════════════════

const P: u64 = 4_294_967_291; // 2^32 - 5

#[derive(Debug, Clone, Copy)]
struct Fp(u32);

impl Fp {
    fn new(val: u32) -> Self { Fp((val as u64 % P) as u32) }

    fn from_i8(val: i8) -> Self {
        Fp(((val as i64).rem_euclid(P as i64)) as u32)
    }

    fn from_i32(val: i32) -> Self {
        Fp(((val as i64).rem_euclid(P as i64)) as u32)
    }

    fn dot_fp_i8(a: &[Fp], b: &[i8]) -> Fp {
        let mut pos: u128 = 0;
        let mut neg: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i16;
            if yi >= 0 { pos += x.0 as u128 * yi as u128; }
            else { neg += x.0 as u128 * (-yi) as u128; }
        }
        let pr = (pos % P as u128) as u64;
        let nr = (neg % P as u128) as u64;
        Fp(if pr >= nr { (pr - nr) as u32 } else { (pr + P - nr) as u32 })
    }

    fn dot_fp_i32(a: &[Fp], b: &[i32]) -> Fp {
        let mut pos: u128 = 0;
        let mut neg: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i64;
            if yi >= 0 { pos += x.0 as u128 * yi as u128; }
            else { neg += x.0 as u128 * (-yi) as u128; }
        }
        let pr = (pos % P as u128) as u64;
        let nr = (neg % P as u128) as u64;
        Fp(if pr >= nr { (pr - nr) as u32 } else { (pr + P - nr) as u32 })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Freivalds
// ═══════════════════════════════════════════════════════════════════════

fn precompute_v(r: &[Fp], w: &[i8], rows: usize, cols: usize) -> Vec<Fp> {
    let mut v = vec![Fp(0); cols];
    for col in 0..cols {
        let mut acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for row in 0..rows {
            let wv = w[row * cols + col] as i16;
            let rv = r[row].0 as u128;
            if wv >= 0 { acc += rv * wv as u128; }
            else { neg_acc += rv * (-wv) as u128; }
        }
        let p = (acc % P as u128) as u64;
        let n = (neg_acc % P as u128) as u64;
        v[col] = if p >= n { Fp((p - n) as u32) } else { Fp((p + P - n) as u32) };
    }
    v
}

fn freivalds_check(v: &[Fp], x: &[i8], r: &[Fp], z: &[i32]) -> bool {
    Fp::dot_fp_i8(v, x).0 == Fp::dot_fp_i32(r, z).0
}

// ═══════════════════════════════════════════════════════════════════════
// Matmul + requantize
// ═══════════════════════════════════════════════════════════════════════

fn matmul_i32(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows).map(|r| (0..cols).map(|c| w[r*cols+c] as i32 * x[c] as i32).sum()).collect()
}

fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// SiLU
// ═══════════════════════════════════════════════════════════════════════

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

fn compute_h_unit_scale(g: &[i32], u: &[i32]) -> Vec<i8> {
    let mut lut = [0.0f32; 256];
    for i in 0..256u16 { lut[i as usize] = silu(i as i8 as f32); }
    g.iter().zip(u.iter()).map(|(&gi, &ui)| {
        let g8 = gi.clamp(-128, 127) as i8;
        let u8v = ui.clamp(-128, 127) as i8;
        let prod = lut[g8 as u8 as usize] * u8v as f32;
        prod.round().clamp(-128.0, 127.0) as i8
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// Merkle (SHA-256, must match Zig domain separators)
// ═══════════════════════════════════════════════════════════════════════

fn hash_pair(l: &[u8; 32], r: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new(); h.update(l); h.update(r); h.finalize().into()
}

fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    let n = leaves.len().next_power_of_two();
    let mut level = vec![[0u8; 32]; n];
    level[..leaves.len()].copy_from_slice(leaves);
    while level.len() > 1 {
        level = level.chunks_exact(2).map(|p| hash_pair(&p[0], &p[1])).collect();
    }
    level[0]
}

/// Hash retained state: domain "vi-retained-v3"
fn hash_retained_state(layers: &[(&[i8], f32)]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vi-retained-v3");
    for (a, scale_a) in layers {
        // i8 as u8 bytes
        let bytes = i8_as_u8_slice(a);
        h.update(bytes);
        h.update(scale_a.to_le_bytes());
        h.update(&[0x00u8]); // no x_attn_i8
    }
    h.finalize().into()
}

/// IO chain: domain "vi-io-v4"
fn io_hash_v4(leaf: [u8; 32], token_id: u32, prev: [u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vi-io-v4");
    h.update(leaf);
    h.update(token_id.to_le_bytes());
    h.update(prev);
    h.finalize().into()
}

fn hash_prompt(prompt: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new(); h.update(b"vi-prompt-v1"); h.update(prompt); h.finalize().into()
}

fn hash_weights(layers: &[LayerWeights]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"vi-weight-chain-v1");
    h.update(b"I8");
    for lw in layers {
        for w in [&lw.wq, &lw.wk, &lw.wv, &lw.wo, &lw.wg, &lw.wu, &lw.wd] {
            let bytes = i8_as_u8_slice(w);
            h.update(bytes);
        }
    }
    h.finalize().into()
}

fn i8_as_u8_slice(s: &[i8]) -> &[u8] {
    unsafe { core::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}

// ═══════════════════════════════════════════════════════════════════════
// Model generation (deterministic LCG, matches Zig's Xoshiro256)
// ═══════════════════════════════════════════════════════════════════════

struct LayerWeights { wq: Vec<i8>, wk: Vec<i8>, wv: Vec<i8>, wo: Vec<i8>, wg: Vec<i8>, wu: Vec<i8>, wd: Vec<i8> }

/// We DON'T match Zig's PRNG — instead we serialize everything the verifier
/// needs, so PRNG compatibility is irrelevant. Both sides agree on the data.
fn lcg_weights(state: &mut u64, n: usize) -> Vec<i8> {
    (0..n).map(|_| {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) & 0xFF) as i8
    }).collect()
}

fn lcg_r_vectors(state: &mut u64, dim: usize) -> Vec<Fp> {
    (0..dim).map(|_| {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        Fp::new((*state >> 32) as u32)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// JSON output types
// ═══════════════════════════════════════════════════════════════════════

#[derive(Serialize)]
struct CrossLangBundle {
    // Config
    config: Config,
    prompt: String,

    // Model weights (layer 0 and layer 1)
    layers: Vec<LayerData>,
    lm_head: Vec<i8>,

    // Verifier key
    r_vectors: Vec<Vec<u32>>,       // r[matrix_type_idx][i]
    v_vectors: Vec<Vec<Vec<u32>>>,  // v[layer][matrix_type_idx][i]

    // Input
    input: Vec<i8>,

    // Traces per layer
    traces: Vec<TraceData>,

    // Merkle
    weight_hash: String,
    retained_leaf_hashes: Vec<String>,  // per-token (here just 1 token)
    merkle_root: String,
    io_chain: Vec<String>,
    prompt_hash: String,

    // Freivalds check results (Rust-side, for cross-validation)
    freivalds_results: Vec<FreivaldsResult>,
}

#[derive(Serialize)]
struct Config {
    hidden_dim: usize, kv_dim: usize, ffn_dim: usize, d_head: usize,
    n_layers: usize, n_q_heads: usize, n_kv_heads: usize, vocab_size: usize,
}

#[derive(Serialize)]
struct LayerData {
    wq: Vec<i8>, wk: Vec<i8>, wv: Vec<i8>, wo: Vec<i8>,
    wg: Vec<i8>, wu: Vec<i8>, wd: Vec<i8>,
}

#[derive(Serialize)]
struct TraceData {
    x_attn: Vec<i8>,
    q: Vec<i32>, k: Vec<i32>, v: Vec<i32>,
    a: Vec<i8>,
    attn_out: Vec<i32>,
    x_ffn: Vec<i8>,
    g: Vec<i32>, u: Vec<i32>,
    h: Vec<i8>,
    ffn_out: Vec<i32>,
}

#[derive(Serialize)]
struct FreivaldsResult {
    layer: usize,
    matrix: String,
    passed: bool,
    lhs: u32,  // v . x mod p
    rhs: u32,  // r . z mod p
}

fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{:02x}", b)).collect()
}

// ═══════════════════════════════════════════════════════════════════════
// Main: generate the full proof bundle
// ═══════════════════════════════════════════════════════════════════════

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "crosslang_bundle.json".into());

    // Config (toy model — matches Zig ModelConfig.toy())
    let hidden_dim = 16;
    let kv_dim = 4;
    let ffn_dim = 32;
    let d_head = 2;
    let n_layers = 2;
    let n_q_heads = 8;
    let n_kv_heads = 2;
    let vocab_size = 64;
    let heads_per_kv = n_q_heads / n_kv_heads;
    let prompt = "Hello, verify me!";

    // Matrix dimensions for each type
    let dims: Vec<(usize, usize)> = vec![
        (hidden_dim, hidden_dim), // Wq
        (kv_dim, hidden_dim),     // Wk
        (kv_dim, hidden_dim),     // Wv
        (hidden_dim, hidden_dim), // Wo
        (ffn_dim, hidden_dim),    // Wg
        (ffn_dim, hidden_dim),    // Wu
        (hidden_dim, ffn_dim),    // Wd
        (vocab_size, hidden_dim), // LmHead
    ];

    // Generate model weights
    let mut lcg = 42u64;
    let mut layers_w = Vec::new();
    for _ in 0..n_layers {
        layers_w.push(LayerWeights {
            wq: lcg_weights(&mut lcg, hidden_dim * hidden_dim),
            wk: lcg_weights(&mut lcg, kv_dim * hidden_dim),
            wv: lcg_weights(&mut lcg, kv_dim * hidden_dim),
            wo: lcg_weights(&mut lcg, hidden_dim * hidden_dim),
            wg: lcg_weights(&mut lcg, ffn_dim * hidden_dim),
            wu: lcg_weights(&mut lcg, ffn_dim * hidden_dim),
            wd: lcg_weights(&mut lcg, hidden_dim * ffn_dim),
        });
    }
    let lm_head = lcg_weights(&mut lcg, vocab_size * hidden_dim);

    // Generate verifier key
    let mut key_lcg = 99u64; // different seed for key
    let mut r_vectors: Vec<Vec<Fp>> = Vec::new();
    for &(out_dim, _) in &dims {
        r_vectors.push(lcg_r_vectors(&mut key_lcg, out_dim));
    }

    // Precompute v = r^T W for each layer
    let mut v_vectors: Vec<Vec<Vec<Fp>>> = Vec::new();
    for layer in &layers_w {
        let weights = [&layer.wq, &layer.wk, &layer.wv, &layer.wo, &layer.wg, &layer.wu, &layer.wd];
        let mut layer_vs = Vec::new();
        for (j, w) in weights.iter().enumerate() {
            let (rows, cols) = dims[j];
            layer_vs.push(precompute_v(&r_vectors[j], w, rows, cols));
        }
        v_vectors.push(layer_vs);
    }

    // Forward pass
    let input: Vec<i8> = (0..hidden_dim as i8).map(|i| i - 8).collect();
    let mut x = input.clone();
    let mut traces = Vec::new();

    for layer in &layers_w {
        let x_attn = x.clone();
        let q = matmul_i32(&layer.wq, &x_attn, hidden_dim, hidden_dim);
        let k = matmul_i32(&layer.wk, &x_attn, kv_dim, hidden_dim);
        let v = matmul_i32(&layer.wv, &x_attn, kv_dim, hidden_dim);

        // Single-token GQA: attn output = V replicated
        let v_i8 = requantize(&v);
        let mut a = vec![0i8; hidden_dim];
        for qh in 0..n_q_heads {
            let kvh = qh / heads_per_kv;
            a[qh*d_head..(qh+1)*d_head].copy_from_slice(&v_i8[kvh*d_head..(kvh+1)*d_head]);
        }

        let attn_out = matmul_i32(&layer.wo, &a, hidden_dim, hidden_dim);
        let x_ffn = requantize(&attn_out);
        let g = matmul_i32(&layer.wg, &x_ffn, ffn_dim, hidden_dim);
        let u = matmul_i32(&layer.wu, &x_ffn, ffn_dim, hidden_dim);
        let h = compute_h_unit_scale(&g, &u);
        let ffn_out = matmul_i32(&layer.wd, &h, hidden_dim, ffn_dim);
        x = requantize(&ffn_out);

        traces.push(TraceData { x_attn, q, k, v, a, attn_out, x_ffn, g, u, h, ffn_out });
    }

    // Freivalds checks (Rust-side)
    let matrix_names = ["Wq", "Wk", "Wv", "Wo", "Wg", "Wu", "Wd"];
    let mut freivalds_results = Vec::new();
    for (li, trace) in traces.iter().enumerate() {
        let inputs_outputs: Vec<(&[i8], &[i32])> = vec![
            (&trace.x_attn, &trace.q),    // Wq
            (&trace.x_attn, &trace.k),    // Wk
            (&trace.x_attn, &trace.v),    // Wv
            (&trace.a, &trace.attn_out),   // Wo
            (&trace.x_ffn, &trace.g),     // Wg
            (&trace.x_ffn, &trace.u),     // Wu
            (&trace.h, &trace.ffn_out),   // Wd
        ];
        for (j, (inp, out)) in inputs_outputs.iter().enumerate() {
            let v = &v_vectors[li][j];
            let r = &r_vectors[j];
            let lhs = Fp::dot_fp_i8(v, inp);
            let rhs = Fp::dot_fp_i32(r, out);
            freivalds_results.push(FreivaldsResult {
                layer: li,
                matrix: matrix_names[j].to_string(),
                passed: lhs.0 == rhs.0,
                lhs: lhs.0,
                rhs: rhs.0,
            });
        }
    }

    // Merkle commitment
    let weight_hash_bytes = hash_weights(&layers_w);

    // Retained state hashes
    let mut retained_hashes = Vec::new();
    for trace in &traces {
        // Build per-layer retained state for this token
        let layer_states: Vec<(&[i8], f32)> = vec![(&trace.a[..], 1.0)];
        // Actually we need ALL layers per token, not per trace
        // For a single token, each trace IS one layer
    }
    // For the toy model single-token case, all layers' `a` go into one hash
    let all_layers: Vec<(&[i8], f32)> = traces.iter().map(|t| (t.a.as_slice(), 1.0f32)).collect();
    let leaf = hash_retained_state(&all_layers);
    retained_hashes.push(hex(&leaf));
    let root = merkle_root(&[leaf]);

    // IO chain
    let prompt_hash_bytes = hash_prompt(prompt.as_bytes());
    let io0 = io_hash_v4(leaf, 0, prompt_hash_bytes);

    // Build JSON bundle
    let bundle = CrossLangBundle {
        config: Config { hidden_dim, kv_dim, ffn_dim, d_head, n_layers, n_q_heads, n_kv_heads, vocab_size },
        prompt: prompt.to_string(),
        layers: layers_w.iter().map(|l| LayerData {
            wq: l.wq.clone(), wk: l.wk.clone(), wv: l.wv.clone(), wo: l.wo.clone(),
            wg: l.wg.clone(), wu: l.wu.clone(), wd: l.wd.clone(),
        }).collect(),
        lm_head,
        r_vectors: r_vectors.iter().map(|rv| rv.iter().map(|f| f.0).collect()).collect(),
        v_vectors: v_vectors.iter().map(|lv| lv.iter().map(|vv| vv.iter().map(|f| f.0).collect()).collect()).collect(),
        input,
        traces: traces.into_iter().map(|t| t).collect(),
        weight_hash: hex(&weight_hash_bytes),
        retained_leaf_hashes: retained_hashes,
        merkle_root: hex(&root),
        io_chain: vec![hex(&io0)],
        prompt_hash: hex(&prompt_hash_bytes),
        freivalds_results,
    };

    let json = serde_json::to_string_pretty(&bundle).unwrap();
    std::fs::write(&out_path, &json).unwrap();
    eprintln!("Wrote cross-language proof bundle to {}", out_path);
    eprintln!("  {} layers, {} Freivalds checks (all passed: {})",
        n_layers,
        bundle.freivalds_results.len(),
        bundle.freivalds_results.iter().all(|r| r.passed),
    );
}
