//! SHA-256 Merkle tree for token trace commitments.
//!
//! The tree commits to per-token hashes. The prover sends the root
//! with the response, then opens individual leaves on challenge.
//! All integer encoding uses little-endian to match the Rust reference.

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Hash = [32]u8;
const zero_hash: Hash = [_]u8{0} ** 32;

pub const MerkleProof = struct {
    leaf_index: u32,
    siblings: []Hash,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MerkleProof) void {
        self.allocator.free(self.siblings);
        self.* = undefined;
    }
};

pub const MerkleTree = struct {
    nodes: []Hash,
    n_leaves: usize,
    padded_size: usize,
    root: Hash,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *MerkleTree) void {
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
};

/// Hash two 32-byte children to produce a parent. H(left || right).
pub fn hashPair(left: *const Hash, right: *const Hash) Hash {
    var h = Sha256.init(.{});
    h.update(left);
    h.update(right);
    return h.finalResult();
}

/// Hash arbitrary data to produce a leaf hash. H(data).
pub fn hashLeaf(data: []const u8) Hash {
    var h = Sha256.init(.{});
    h.update(data);
    return h.finalResult();
}

/// Encode u32 as 4 little-endian bytes (matches Rust .to_le_bytes()).
inline fn leU32(val: u32) [4]u8 {
    return std.mem.toBytes(std.mem.nativeToLittle(u32, val));
}

/// Build a Merkle tree from leaf hashes. Pads to next power of 2.
pub fn buildTree(allocator: std.mem.Allocator, leaves: []const Hash) !MerkleTree {
    if (leaves.len == 0) return error.EmptyLeaves;

    const n = std.math.ceilPowerOfTwo(usize, leaves.len) catch unreachable;
    const total_nodes = 2 * n - 1;
    const nodes = try allocator.alloc(Hash, total_nodes);
    errdefer allocator.free(nodes);

    // Level 0: leaves padded with zero hashes.
    @memcpy(nodes[0..leaves.len], leaves);
    for (leaves.len..n) |i| {
        nodes[i] = zero_hash;
    }

    // Build parent levels.
    var level_start: usize = 0;
    var level_size: usize = n;
    var write_pos: usize = n;
    while (level_size > 1) {
        var i: usize = 0;
        while (i < level_size) : (i += 2) {
            nodes[write_pos] = hashPair(&nodes[level_start + i], &nodes[level_start + i + 1]);
            write_pos += 1;
        }
        level_start += level_size;
        level_size /= 2;
    }

    return .{
        .nodes = nodes,
        .n_leaves = leaves.len,
        .padded_size = n,
        .root = nodes[total_nodes - 1],
        .allocator = allocator,
    };
}

/// Compute Merkle root without storing intermediate nodes. O(log N) stack.
pub fn computeRoot(allocator: std.mem.Allocator, leaves: []const Hash) !Hash {
    if (leaves.len == 0) return error.EmptyLeaves;

    const n = std.math.ceilPowerOfTwo(usize, leaves.len) catch unreachable;
    var level = try allocator.alloc(Hash, n);
    defer allocator.free(level);

    @memcpy(level[0..leaves.len], leaves);
    for (leaves.len..n) |i| {
        level[i] = zero_hash;
    }

    var size = n;
    while (size > 1) {
        var i: usize = 0;
        var w: usize = 0;
        while (i < size) : ({
            i += 2;
            w += 1;
        }) {
            level[w] = hashPair(&level[i], &level[i + 1]);
        }
        size /= 2;
    }

    return level[0];
}

/// Generate a Merkle proof for the leaf at `index`.
pub fn prove(allocator: std.mem.Allocator, tree: *const MerkleTree, index: usize) !MerkleProof {
    if (index >= tree.n_leaves) return error.IndexOutOfRange;

    // For padded_size == 1, depth is 0 and proof has no siblings.
    const depth = if (tree.padded_size <= 1) 0 else std.math.log2(tree.padded_size);
    const siblings = try allocator.alloc(Hash, depth);
    errdefer allocator.free(siblings);

    var idx = index;
    var level_start: usize = 0;
    var level_size: usize = tree.padded_size;
    var sib_idx: usize = 0;

    while (level_size > 1) {
        const sibling = if (idx % 2 == 0) idx + 1 else idx - 1;
        siblings[sib_idx] = tree.nodes[level_start + sibling];
        sib_idx += 1;
        level_start += level_size;
        level_size /= 2;
        idx /= 2;
    }

    return .{
        .leaf_index = @intCast(index),
        .siblings = siblings,
        .allocator = allocator,
    };
}

/// Verify a Merkle proof against a known root.
pub fn verify(root: *const Hash, leaf: *const Hash, proof_obj: *const MerkleProof) bool {
    var current = leaf.*;
    var idx: usize = proof_obj.leaf_index;

    for (proof_obj.siblings) |sibling| {
        current = if (idx % 2 == 0)
            hashPair(&current, &sibling)
        else
            hashPair(&sibling, &current);
        idx /= 2;
    }

    return std.mem.eql(u8, &current, root);
}

// ═══════════════════════════════════════════════════════════════════════
// Domain-separated hashing (must match Rust byte-for-byte)
// ═══════════════════════════════════════════════════════════════════════

/// Input for retained state hashing.
pub const RetainedLayerInput = struct {
    a: []const i8,
    scale_a: f32,
    x_attn_i8: ?[]const i8 = null,
    scale_x_attn: ?f32 = null,
};

/// Hash retained token state for trace commitment.
/// Domain: "vi-retained-v3".
pub fn hashRetainedStateDirect(layers: []const RetainedLayerInput) Hash {
    var h = Sha256.init(.{});
    h.update("vi-retained-v3");

    for (layers) |ls| {
        h.update(std.mem.sliceAsBytes(ls.a));
        h.update(&std.mem.toBytes(ls.scale_a));
        if (ls.x_attn_i8) |xa| {
            h.update(&[_]u8{0x01});
            h.update(std.mem.sliceAsBytes(xa));
            h.update(&std.mem.toBytes(ls.scale_x_attn.?));
        } else {
            h.update(&[_]u8{0x00});
        }
    }

    return h.finalResult();
}

/// Hash retained state with optional final residual binding.
/// Domain: "vi-retained-fr-v1" when residual is present.
pub fn hashRetainedWithResidual(layers: []const RetainedLayerInput, final_residual: ?[]const f32) Hash {
    const base = hashRetainedStateDirect(layers);
    if (final_residual) |fr| {
        var h = Sha256.init(.{});
        h.update("vi-retained-fr-v1");
        h.update(&base);
        h.update(std.mem.sliceAsBytes(fr));
        return h.finalResult();
    }
    return base;
}

/// V4 IO chain hash. Domain: "vi-io-v4".
///   io_t = H("vi-io-v4" || leaf_hash || token_id_LE || prev_io)
pub fn ioHashV4(leaf_hash: Hash, token_id: u32, prev_io_hash: Hash) Hash {
    var h = Sha256.init(.{});
    h.update("vi-io-v4");
    h.update(&leaf_hash);
    h.update(&leU32(token_id));
    h.update(&prev_io_hash);
    return h.finalResult();
}

/// IO chain genesis hash. Domain: "vi-io-genesis-v4".
pub fn ioGenesisV4(prompt_hash: Hash) Hash {
    var h = Sha256.init(.{});
    h.update("vi-io-genesis-v4");
    h.update(&prompt_hash);
    return h.finalResult();
}

/// Hash a KV entry. Domain: "vi-kv-v1".
///   H("vi-kv-v1" || layer_LE || position_LE || k_roped_bytes || v_deq_bytes)
pub fn hashKvEntry(layer: u32, position: u32, k_roped: []const f64, v_deq: []const f64) Hash {
    var h = Sha256.init(.{});
    h.update("vi-kv-v1");
    h.update(&leU32(layer));
    h.update(&leU32(position));
    h.update(std.mem.sliceAsBytes(k_roped));
    h.update(std.mem.sliceAsBytes(v_deq));
    return h.finalResult();
}

/// Hash an embedding table row. Domain: "vi-embedding-v1".
pub fn hashEmbeddingRow(row: []const f32) Hash {
    var h = Sha256.init(.{});
    h.update("vi-embedding-v1");
    h.update(std.mem.sliceAsBytes(row));
    return h.finalResult();
}

/// Hash a seed for commitment. Domain: "vi-seed-v1".
pub fn hashSeed(seed: []const u8) Hash {
    var h = Sha256.init(.{});
    h.update("vi-seed-v1");
    h.update(seed);
    return h.finalResult();
}

/// Hash a prompt. Domain: "vi-prompt-v1".
pub fn hashPrompt(prompt_bytes: []const u8) Hash {
    var h = Sha256.init(.{});
    h.update("vi-prompt-v1");
    h.update(prompt_bytes);
    return h.finalResult();
}

/// Compute canonical weight-chain hash. Domain: "vi-weight-chain-v1".
///
/// H("vi-weight-chain-v1" || source_dtype || n_layers || per-matrix-scales || all weights)
pub fn hashWeights(
    source_dtype: []const u8,
    n_layers: usize,
    quant_scales: []const u8,
    weight_data: []const u8,
) Hash {
    var h = Sha256.init(.{});
    h.update("vi-weight-chain-v1");
    h.update(source_dtype);
    h.update(&leU32(@intCast(n_layers)));
    h.update(quant_scales);
    h.update(weight_data);
    return h.finalResult();
}

/// Derive challenge indices from commitment root, revealed seed, and counter.
///   index = SHA256(root || seed || counter) % n_tokens
pub fn deriveChallenge(root: Hash, seed: []const u8, counter: u32, n_tokens: u32) u32 {
    var h = Sha256.init(.{});
    h.update(&root);
    h.update(seed);
    h.update(&leU32(counter));
    const digest = h.finalResult();
    const raw = std.mem.readInt(u32, digest[0..4], .little);
    return raw % n_tokens;
}

// --- Spec hashing ---

fn hashOptional32(h: *Sha256, val: ?*const Hash) void {
    if (val) |v| {
        h.update(&[_]u8{0x01});
        h.update(v);
    } else {
        h.update(&[_]u8{0x00});
    }
}

fn hashOptionalString(h: *Sha256, val: ?[]const u8) void {
    if (val) |v| {
        h.update(&[_]u8{0x01});
        h.update(&leU32(@intCast(v.len)));
        h.update(v);
    } else {
        h.update(&[_]u8{0x00});
    }
}

const types = @import("types.zig");

/// Hash an InputSpec. Domain: "vi-input-v1".
pub fn hashInputSpec(spec: *const types.InputSpec) Hash {
    var h = Sha256.init(.{});
    h.update("vi-input-v1");
    h.update(&spec.tokenizer_hash);
    hashOptional32(&h, if (spec.system_prompt_hash != null) &spec.system_prompt_hash.? else null);
    hashOptional32(&h, if (spec.chat_template_hash != null) &spec.chat_template_hash.? else null);
    hashOptionalString(&h, spec.bos_eos_policy);
    hashOptionalString(&h, spec.truncation_policy);
    hashOptionalString(&h, spec.special_token_policy);
    hashOptionalString(&h, spec.padding_policy);
    return h.finalResult();
}

/// Hash a ModelSpec. Domain: "vi-model-v1".
pub fn hashModelSpec(spec: *const types.ModelSpec) Hash {
    var h = Sha256.init(.{});
    h.update("vi-model-v1");
    hashOptional32(&h, if (spec.weight_hash != null) &spec.weight_hash.? else null);
    hashOptional32(&h, if (spec.quant_hash != null) &spec.quant_hash.? else null);
    hashOptional32(&h, if (spec.rope_config_hash != null) &spec.rope_config_hash.? else null);
    if (spec.rmsnorm_eps) |eps| {
        h.update(&[_]u8{0x01});
        h.update(&std.mem.toBytes(eps));
    } else {
        h.update(&[_]u8{0x00});
    }
    hashOptional32(&h, if (spec.adapter_hash != null) &spec.adapter_hash.? else null);
    return h.finalResult();
}

/// Hash a DeploymentManifest. Domain: "vi-manifest-v4".
pub fn hashManifest(manifest: *const types.DeploymentManifest) Hash {
    var h = Sha256.init(.{});
    h.update("vi-manifest-v4");
    h.update(&manifest.input_spec_hash);
    h.update(&manifest.model_spec_hash);
    h.update(&manifest.decode_spec_hash);
    h.update(&manifest.output_spec_hash);
    return h.finalResult();
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "merkle_single_leaf" {
    const allocator = testing.allocator;
    const leaf = hashLeaf("hello");
    const leaves = [_]Hash{leaf};
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    // Single leaf: padded_size=1, root == leaf.
    try testing.expectEqualSlices(u8, &leaf, &tree.root);

    // Prove and verify.
    var prf = try prove(allocator, &tree, 0);
    defer prf.deinit();
    try testing.expect(verify(&tree.root, &leaf, &prf));
}

test "merkle_two_leaves" {
    const allocator = testing.allocator;
    const leaf0 = hashLeaf("hello");
    const leaf1 = hashLeaf("world");
    const leaves = [_]Hash{ leaf0, leaf1 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    const expected = hashPair(&leaf0, &leaf1);
    try testing.expectEqualSlices(u8, &expected, &tree.root);
}

test "merkle_prove_verify_roundtrip" {
    const allocator = testing.allocator;
    const leaf0 = hashLeaf("a");
    const leaf1 = hashLeaf("b");
    const leaf2 = hashLeaf("c");
    const leaves = [_]Hash{ leaf0, leaf1, leaf2 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    for (0..3) |i| {
        var prf = try prove(allocator, &tree, i);
        defer prf.deinit();
        try testing.expect(verify(&tree.root, &leaves[i], &prf));
    }
}

test "merkle_wrong_leaf_fails" {
    const allocator = testing.allocator;
    const leaf0 = hashLeaf("a");
    const leaf1 = hashLeaf("b");
    const leaves = [_]Hash{ leaf0, leaf1 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    var prf = try prove(allocator, &tree, 0);
    defer prf.deinit();

    const wrong = hashLeaf("wrong");
    try testing.expect(!verify(&tree.root, &wrong, &prf));
}

test "compute_root_matches_build_tree" {
    const allocator = testing.allocator;
    const leaves = [_]Hash{
        hashLeaf("a"), hashLeaf("b"), hashLeaf("c"),
        hashLeaf("d"), hashLeaf("e"),
    };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    const root = try computeRoot(allocator, &leaves);
    try testing.expectEqualSlices(u8, &tree.root, &root);
}

test "derive_challenge_deterministic" {
    const root = hashLeaf("root");
    const seed = "test_seed";
    const c0 = deriveChallenge(root, seed, 0, 100);
    const c1 = deriveChallenge(root, seed, 0, 100);
    try testing.expectEqual(c0, c1);

    // Different counter → different challenge.
    const c2 = deriveChallenge(root, seed, 1, 100);
    try testing.expect(c0 != c2);
}

test "io_chain_ordering" {
    const prompt_hash = hashPrompt("test prompt");
    const leaf0 = hashLeaf("token0");
    const leaf1 = hashLeaf("token1");

    const io0 = ioHashV4(leaf0, 42, prompt_hash);
    const io1 = ioHashV4(leaf1, 43, io0);

    // Deterministic.
    const io0_again = ioHashV4(leaf0, 42, prompt_hash);
    try testing.expectEqualSlices(u8, &io0, &io0_again);

    // Different inputs → different hashes.
    try testing.expect(!std.mem.eql(u8, &io0, &io1));
}

test "hash_retained_with_residual" {
    const a = [_]i8{ 1, 2, 3 };
    const layers = [_]RetainedLayerInput{.{ .a = &a, .scale_a = 1.0 }};
    const base = hashRetainedStateDirect(&layers);
    const with_residual = hashRetainedWithResidual(&layers, &[_]f32{ 1.0, 2.0 });
    const without_residual = hashRetainedWithResidual(&layers, null);

    // Without residual should equal base.
    try testing.expectEqualSlices(u8, &base, &without_residual);
    // With residual should differ.
    try testing.expect(!std.mem.eql(u8, &base, &with_residual));
}
