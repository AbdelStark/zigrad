//! Generate deterministic test vectors for differential testing
//! of the Zig CommitLLM implementation against Rust.
//!
//! Outputs JSON files that the Zig tests can load and verify.

use serde::Serialize;
use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Field arithmetic (inline, matching verilm-core/field.rs exactly)
// ---------------------------------------------------------------------------

const P: u64 = 4_294_967_291; // 2^32 - 5

#[derive(Debug, Clone, Copy)]
struct Fp(u32);

impl Fp {
    fn new(val: u32) -> Self {
        Fp((val as u64 % P) as u32)
    }

    fn from_i8(val: i8) -> Self {
        let v = val as i64;
        let reduced = v.rem_euclid(P as i64) as u32;
        Fp(reduced)
    }

    fn from_i32(val: i32) -> Self {
        let v = val as i64;
        let reduced = v.rem_euclid(P as i64) as u32;
        Fp(reduced)
    }

    fn add(self, other: Self) -> Self {
        let sum = self.0 as u64 + other.0 as u64;
        Fp((sum % P) as u32)
    }

    fn sub(self, other: Self) -> Self {
        let diff = self.0 as u64 + P - other.0 as u64;
        Fp((diff % P) as u32)
    }

    fn mul(self, other: Self) -> Self {
        let prod = self.0 as u64 * other.0 as u64;
        Fp((prod % P) as u32)
    }

    fn dot(a: &[Fp], b: &[Fp]) -> Fp {
        let mut acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            acc += x.0 as u128 * y.0 as u128;
        }
        Fp((acc % P as u128) as u32)
    }

    fn dot_fp_i8(a: &[Fp], b: &[i8]) -> Fp {
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i16;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
        }
        let pos_reduced = (pos_acc % P as u128) as u64;
        let neg_reduced = (neg_acc % P as u128) as u64;
        if pos_reduced >= neg_reduced {
            Fp((pos_reduced - neg_reduced) as u32)
        } else {
            Fp((pos_reduced + P - neg_reduced) as u32)
        }
    }

    fn dot_fp_i32(a: &[Fp], b: &[i32]) -> Fp {
        let mut pos_acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for (x, y) in a.iter().zip(b.iter()) {
            let yi = *y as i64;
            if yi >= 0 {
                pos_acc += x.0 as u128 * yi as u128;
            } else {
                neg_acc += x.0 as u128 * (-yi) as u128;
            }
        }
        let pos_reduced = (pos_acc % P as u128) as u64;
        let neg_reduced = (neg_acc % P as u128) as u64;
        if pos_reduced >= neg_reduced {
            Fp((pos_reduced - neg_reduced) as u32)
        } else {
            Fp((pos_reduced + P - neg_reduced) as u32)
        }
    }
}

// ---------------------------------------------------------------------------
// Freivalds
// ---------------------------------------------------------------------------

fn precompute_v(r: &[Fp], weight: &[i8], rows: usize, cols: usize) -> Vec<Fp> {
    let mut v = vec![Fp(0); cols];
    for col in 0..cols {
        let mut acc: u128 = 0;
        let mut neg_acc: u128 = 0;
        for row in 0..rows {
            let w_val = weight[row * cols + col] as i16;
            let r_val = r[row].0 as u128;
            if w_val >= 0 {
                acc += r_val * w_val as u128;
            } else {
                neg_acc += r_val * (-w_val) as u128;
            }
        }
        let p128 = P as u128;
        let pos = (acc % p128) as u64;
        let neg = (neg_acc % p128) as u64;
        v[col] = if pos >= neg {
            Fp((pos - neg) as u32)
        } else {
            Fp((pos + P - neg) as u32)
        };
    }
    v
}

// ---------------------------------------------------------------------------
// Merkle
// ---------------------------------------------------------------------------

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    let n = leaves.len().next_power_of_two();
    let zero = [0u8; 32];

    let mut level: Vec<[u8; 32]> = Vec::with_capacity(n);
    level.extend_from_slice(leaves);
    level.resize(n, zero);

    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len() / 2);
        for pair in level.chunks_exact(2) {
            next.push(hash_pair(&pair[0], &pair[1]));
        }
        level = next;
    }

    level[0]
}

// ---------------------------------------------------------------------------
// SiLU
// ---------------------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn build_silu_lut(scale: f32) -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for i in 0..256u16 {
        let g = i as i8;
        let x = g as f32 * scale;
        lut[i as usize] = silu(x);
    }
    lut
}

fn compute_h_unit_scale(g_acc: &[i32], u_acc: &[i32]) -> Vec<i8> {
    let lut = build_silu_lut(1.0);
    g_acc.iter().zip(u_acc.iter()).map(|(&g, &u)| {
        let g_i8 = g.clamp(-128, 127) as i8;
        let u_i8 = u.clamp(-128, 127) as i8;
        let silu_g = lut[g_i8 as u8 as usize];
        let product = silu_g * u_i8 as f32;
        product.round().clamp(-128.0, 127.0) as i8
    }).collect()
}

// ---------------------------------------------------------------------------
// Matmul
// ---------------------------------------------------------------------------

fn matmul_i32(w: &[i8], x: &[i8], rows: usize, cols: usize) -> Vec<i32> {
    (0..rows).map(|r| {
        (0..cols).map(|c| w[r * cols + c] as i32 * x[c] as i32).sum()
    }).collect()
}

fn requantize(acc: &[i32]) -> Vec<i8> {
    acc.iter().map(|&v| v.clamp(-128, 127) as i8).collect()
}

// ---------------------------------------------------------------------------
// JSON output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct FieldVectors {
    from_i8: Vec<FromI8Case>,
    from_i32: Vec<FromI32Case>,
    dot_fp_fp: Vec<DotCase>,
    dot_fp_i8: Vec<DotI8Case>,
    dot_fp_i32: Vec<DotI32Case>,
    add_sub_mul: Vec<ArithCase>,
}

#[derive(Serialize)]
struct FromI8Case { input: i8, expected: u32 }
#[derive(Serialize)]
struct FromI32Case { input: i32, expected: u32 }
#[derive(Serialize)]
struct DotCase { a: Vec<u32>, b: Vec<u32>, expected: u32 }
#[derive(Serialize)]
struct DotI8Case { a: Vec<u32>, b: Vec<i8>, expected: u32 }
#[derive(Serialize)]
struct DotI32Case { a: Vec<u32>, b: Vec<i32>, expected: u32 }
#[derive(Serialize)]
struct ArithCase { a: u32, b: u32, add: u32, sub: u32, mul: u32 }

#[derive(Serialize)]
struct FreivaldsVectors {
    cases: Vec<FreivaldsCase>,
}

#[derive(Serialize)]
struct FreivaldsCase {
    rows: usize,
    cols: usize,
    r: Vec<u32>,
    weight: Vec<i8>,
    x: Vec<i8>,
    z: Vec<i32>,
    v: Vec<u32>,
    check_passes: bool,
}

#[derive(Serialize)]
struct MerkleVectors {
    cases: Vec<MerkleCase>,
}

#[derive(Serialize)]
struct MerkleCase {
    leaves: Vec<String>,
    root: String,
}

#[derive(Serialize)]
struct SiluVectors {
    lut_unit: Vec<f32>,
    h_cases: Vec<HCase>,
}

#[derive(Serialize)]
struct HCase {
    g_acc: Vec<i32>,
    u_acc: Vec<i32>,
    h: Vec<i8>,
}

#[derive(Serialize)]
struct E2EVectors {
    config: E2EConfig,
    model_layer0_wq: Vec<i8>,
    input: Vec<i8>,
    layer0_q: Vec<i32>,
    layer0_k: Vec<i32>,
    layer0_v: Vec<i32>,
    layer0_a: Vec<i8>,
    layer0_attn_out: Vec<i32>,
    layer0_x_ffn: Vec<i8>,
    layer0_g: Vec<i32>,
    layer0_u: Vec<i32>,
    layer0_h: Vec<i8>,
    layer0_ffn_out: Vec<i32>,
    weight_hash: String,
}

#[derive(Serialize)]
struct E2EConfig {
    hidden_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
    d_head: usize,
    n_layers: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
}

fn main() {
    let out_dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());

    // --- Field vectors ---
    let field_vecs = generate_field_vectors();
    write_json(&format!("{}/field_vectors.json", out_dir), &field_vecs);

    // --- Freivalds vectors ---
    let frei_vecs = generate_freivalds_vectors();
    write_json(&format!("{}/freivalds_vectors.json", out_dir), &frei_vecs);

    // --- Merkle vectors ---
    let merkle_vecs = generate_merkle_vectors();
    write_json(&format!("{}/merkle_vectors.json", out_dir), &merkle_vecs);

    // --- SiLU vectors ---
    let silu_vecs = generate_silu_vectors();
    write_json(&format!("{}/silu_vectors.json", out_dir), &silu_vecs);

    // --- E2E vectors ---
    let e2e_vecs = generate_e2e_vectors();
    write_json(&format!("{}/e2e_vectors.json", out_dir), &e2e_vecs);

    eprintln!("Generated all test vectors in {}", out_dir);
}

fn write_json<T: Serialize>(path: &str, data: &T) {
    let json = serde_json::to_string_pretty(data).unwrap();
    std::fs::write(path, json).unwrap();
    eprintln!("  Wrote {}", path);
}

fn generate_field_vectors() -> FieldVectors {
    let from_i8: Vec<FromI8Case> = vec![-128, -1, 0, 1, 127, -64, 42]
        .into_iter()
        .map(|v| FromI8Case { input: v, expected: Fp::from_i8(v).0 })
        .collect();

    let from_i32: Vec<FromI32Case> = vec![-2_000_000_000, -1, 0, 1, 2_000_000_000, i32::MIN, i32::MAX]
        .into_iter()
        .map(|v| FromI32Case { input: v, expected: Fp::from_i32(v).0 })
        .collect();

    let dot_cases = vec![
        (vec![1, 2, 3], vec![4, 5, 6]),
        (vec![P as u32 - 1, 1], vec![2, P as u32 - 1]),
        (vec![0, 0, 0], vec![1, 2, 3]),
    ];
    let dot_fp_fp: Vec<DotCase> = dot_cases.into_iter().map(|(a, b)| {
        let a_fp: Vec<Fp> = a.iter().map(|&v| Fp::new(v)).collect();
        let b_fp: Vec<Fp> = b.iter().map(|&v| Fp::new(v)).collect();
        DotCase { a: a.clone(), b: b.clone(), expected: Fp::dot(&a_fp, &b_fp).0 }
    }).collect();

    let dot_i8_cases = vec![
        (vec![10u32, 20, 30], vec![7i8, 8, -1]),
        (vec![100, 200], vec![-128, 127]),
        (vec![P as u32 - 1], vec![-1i8]),
    ];
    let dot_fp_i8: Vec<DotI8Case> = dot_i8_cases.into_iter().map(|(a, b)| {
        let a_fp: Vec<Fp> = a.iter().map(|&v| Fp::new(v)).collect();
        DotI8Case { a: a.clone(), b: b.clone(), expected: Fp::dot_fp_i8(&a_fp, &b).0 }
    }).collect();

    let dot_i32_cases = vec![
        (vec![10u32, 20, 30], vec![23i32, 53, 83]),
        (vec![1], vec![-100_000]),
        (vec![100, 200], vec![i32::MAX, i32::MIN]),
    ];
    let dot_fp_i32: Vec<DotI32Case> = dot_i32_cases.into_iter().map(|(a, b)| {
        let a_fp: Vec<Fp> = a.iter().map(|&v| Fp::new(v)).collect();
        DotI32Case { a: a.clone(), b: b.clone(), expected: Fp::dot_fp_i32(&a_fp, &b).0 }
    }).collect();

    let arith_pairs = vec![
        (0u32, 0u32), (1, 1), (P as u32 - 1, 1), (P as u32 - 1, P as u32 - 1),
        (1_000_000, 2_000_000), (42, 99),
    ];
    let add_sub_mul: Vec<ArithCase> = arith_pairs.into_iter().map(|(a, b)| {
        let fa = Fp::new(a);
        let fb = Fp::new(b);
        ArithCase { a, b, add: fa.add(fb).0, sub: fa.sub(fb).0, mul: fa.mul(fb).0 }
    }).collect();

    FieldVectors { from_i8, from_i32, dot_fp_fp, dot_fp_i8, dot_fp_i32, add_sub_mul }
}

fn generate_freivalds_vectors() -> FreivaldsVectors {
    let mut cases = Vec::new();

    // Case 1: 3x2 correct
    {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp(10), Fp(20), Fp(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 83];
        let v = precompute_v(&r, &w, 3, 2);
        cases.push(FreivaldsCase {
            rows: 3, cols: 2,
            r: r.iter().map(|f| f.0).collect(),
            weight: w, x, z,
            v: v.iter().map(|f| f.0).collect(),
            check_passes: true,
        });
    }

    // Case 2: 3x2 wrong
    {
        let w: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let r = vec![Fp(10), Fp(20), Fp(30)];
        let x: Vec<i8> = vec![7, 8];
        let z: Vec<i32> = vec![23, 53, 84]; // 84 != 83
        let v = precompute_v(&r, &w, 3, 2);
        cases.push(FreivaldsCase {
            rows: 3, cols: 2,
            r: r.iter().map(|f| f.0).collect(),
            weight: w, x, z,
            v: v.iter().map(|f| f.0).collect(),
            check_passes: false,
        });
    }

    // Case 3: negative weights
    {
        let w: Vec<i8> = vec![-1, 2, 3, -4];
        let r = vec![Fp(5), Fp(10)];
        let x: Vec<i8> = vec![3, 7];
        let z: Vec<i32> = vec![11, -19];
        let v = precompute_v(&r, &w, 2, 2);
        cases.push(FreivaldsCase {
            rows: 2, cols: 2,
            r: r.iter().map(|f| f.0).collect(),
            weight: w, x, z,
            v: v.iter().map(|f| f.0).collect(),
            check_passes: true,
        });
    }

    // Case 4: identity 3x3
    {
        let w: Vec<i8> = vec![1, 0, 0, 0, 1, 0, 0, 0, 1];
        let r = vec![Fp(42), Fp(99), Fp(7)];
        let x: Vec<i8> = vec![10, 20, 30];
        let z: Vec<i32> = vec![10, 20, 30];
        let v = precompute_v(&r, &w, 3, 3);
        cases.push(FreivaldsCase {
            rows: 3, cols: 3,
            r: r.iter().map(|f| f.0).collect(),
            weight: w, x, z,
            v: v.iter().map(|f| f.0).collect(),
            check_passes: true,
        });
    }

    // Case 5: larger 4x4 with mixed signs
    {
        let w: Vec<i8> = vec![
            10, -20, 30, -40,
            -50, 60, -70, 80,
            90, -100, 110, -120,
            -1, 2, -3, 4,
        ];
        let r = vec![Fp(1000), Fp(2000), Fp(3000), Fp(4000)];
        let x: Vec<i8> = vec![5, -10, 15, -20];
        let z = matmul_i32(&w, &x, 4, 4);
        let v = precompute_v(&r, &w, 4, 4);
        cases.push(FreivaldsCase {
            rows: 4, cols: 4,
            r: r.iter().map(|f| f.0).collect(),
            weight: w, x, z,
            v: v.iter().map(|f| f.0).collect(),
            check_passes: true,
        });
    }

    FreivaldsVectors { cases }
}

fn generate_merkle_vectors() -> MerkleVectors {
    let mut cases = Vec::new();

    // 1 leaf
    {
        let leaf = hash_leaf(b"hello");
        let root = merkle_root(&[leaf]);
        cases.push(MerkleCase {
            leaves: vec![hex::encode(leaf)],
            root: hex::encode(root),
        });
    }

    // 2 leaves
    {
        let leaf0 = hash_leaf(b"hello");
        let leaf1 = hash_leaf(b"world");
        let root = merkle_root(&[leaf0, leaf1]);
        cases.push(MerkleCase {
            leaves: vec![hex::encode(leaf0), hex::encode(leaf1)],
            root: hex::encode(root),
        });
    }

    // 4 leaves
    {
        let leaves: Vec<[u8; 32]> = (0..4).map(|i| hash_leaf(format!("leaf{}", i).as_bytes())).collect();
        let root = merkle_root(&leaves);
        cases.push(MerkleCase {
            leaves: leaves.iter().map(hex::encode).collect(),
            root: hex::encode(root),
        });
    }

    // 5 leaves (non-power-of-2)
    {
        let leaves: Vec<[u8; 32]> = (0..5).map(|i| hash_leaf(format!("leaf{}", i).as_bytes())).collect();
        let root = merkle_root(&leaves);
        cases.push(MerkleCase {
            leaves: leaves.iter().map(hex::encode).collect(),
            root: hex::encode(root),
        });
    }

    MerkleVectors { cases }
}

fn generate_silu_vectors() -> SiluVectors {
    let lut = build_silu_lut(1.0);

    let h_cases = vec![
        HCase {
            g_acc: vec![10, -5, 127, -128, 200],
            u_acc: vec![20, 30, -10, 50, -60],
            h: compute_h_unit_scale(&[10, -5, 127, -128, 200], &[20, 30, -10, 50, -60]),
        },
        HCase {
            g_acc: vec![0, 0, 0],
            u_acc: vec![100, -100, 0],
            h: compute_h_unit_scale(&[0, 0, 0], &[100, -100, 0]),
        },
        HCase {
            g_acc: vec![1000, -1000],
            u_acc: vec![1000, -1000],
            h: compute_h_unit_scale(&[1000, -1000], &[1000, -1000]),
        },
    ];

    SiluVectors { lut_unit: lut.to_vec(), h_cases }
}

fn generate_e2e_vectors() -> E2EVectors {
    // Use a simple deterministic model (not ChaCha20 — we generate weights manually)
    let hidden_dim = 16;
    let kv_dim = 4;
    let ffn_dim = 32;
    let d_head = 2;
    let n_q_heads = 8;
    let n_kv_heads = 2;
    let heads_per_kv = n_q_heads / n_kv_heads;

    // Generate deterministic weights using a simple LCG
    let mut lcg_state: u64 = 42;
    let mut next_i8 = || -> i8 {
        lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((lcg_state >> 33) & 0xFF) as i8
    };

    let wq: Vec<i8> = (0..hidden_dim * hidden_dim).map(|_| next_i8()).collect();
    let wk: Vec<i8> = (0..kv_dim * hidden_dim).map(|_| next_i8()).collect();
    let wv: Vec<i8> = (0..kv_dim * hidden_dim).map(|_| next_i8()).collect();
    let wo: Vec<i8> = (0..hidden_dim * hidden_dim).map(|_| next_i8()).collect();
    let wg: Vec<i8> = (0..ffn_dim * hidden_dim).map(|_| next_i8()).collect();
    let wu: Vec<i8> = (0..ffn_dim * hidden_dim).map(|_| next_i8()).collect();
    let wd: Vec<i8> = (0..hidden_dim * ffn_dim).map(|_| next_i8()).collect();

    // Input
    let input: Vec<i8> = (0..hidden_dim as i8).map(|i| i - 8).collect();

    // Forward pass layer 0
    let q = matmul_i32(&wq, &input, hidden_dim, hidden_dim);
    let k = matmul_i32(&wk, &input, kv_dim, hidden_dim);
    let v = matmul_i32(&wv, &input, kv_dim, hidden_dim);

    // GQA: single token, softmax = 1.0, attn_out = V replicated
    let v_i8 = requantize(&v);
    let mut a = vec![0i8; hidden_dim];
    for qh in 0..n_q_heads {
        let kv_head = qh / heads_per_kv;
        let src_start = kv_head * d_head;
        let dst_start = qh * d_head;
        a[dst_start..dst_start + d_head].copy_from_slice(&v_i8[src_start..src_start + d_head]);
    }

    let attn_out = matmul_i32(&wo, &a, hidden_dim, hidden_dim);
    let x_ffn = requantize(&attn_out);

    let g = matmul_i32(&wg, &x_ffn, ffn_dim, hidden_dim);
    let u = matmul_i32(&wu, &x_ffn, ffn_dim, hidden_dim);
    let h = compute_h_unit_scale(&g, &u);
    let ffn_out = matmul_i32(&wd, &h, hidden_dim, ffn_dim);

    // Weight hash
    let mut hasher = Sha256::new();
    hasher.update(unsafe { std::slice::from_raw_parts(wq.as_ptr() as *const u8, wq.len()) });
    // Only hash Wq for the test vector (simplified)
    let weight_hash = hex::encode(hasher.finalize());

    E2EVectors {
        config: E2EConfig {
            hidden_dim, kv_dim, ffn_dim, d_head, n_layers: 2, n_q_heads, n_kv_heads, vocab_size: 64,
        },
        model_layer0_wq: wq,
        input,
        layer0_q: q,
        layer0_k: k,
        layer0_v: v,
        layer0_a: a,
        layer0_attn_out: attn_out,
        layer0_x_ffn: x_ffn,
        layer0_g: g,
        layer0_u: u,
        layer0_h: h,
        layer0_ffn_out: ffn_out,
        weight_hash,
    }
}

mod hex {
    pub fn encode(data: impl AsRef<[u8]>) -> String {
        data.as_ref().iter().map(|b| format!("{:02x}", b)).collect()
    }
}
