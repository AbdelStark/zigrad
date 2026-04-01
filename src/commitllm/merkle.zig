//! SHA-256 Merkle tree for token trace commitments.
//!
//! The tree commits to per-token hashes. The prover sends the root
//! with the response, then opens individual leaves on challenge.

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

/// Hash two 32-byte nodes to produce a parent hash.
pub fn hashPair(left: *const Hash, right: *const Hash) Hash {
    var h = Sha256.init(.{});
    h.update(left);
    h.update(right);
    return h.finalResult();
}

/// Hash arbitrary data to produce a leaf hash.
pub fn hashLeaf(data: []const u8) Hash {
    var h = Sha256.init(.{});
    h.update(data);
    return h.finalResult();
}

/// Build a Merkle tree from leaf hashes. Returns all nodes and root.
pub fn buildTree(allocator: std.mem.Allocator, leaves: []const Hash) !MerkleTree {
    std.debug.assert(leaves.len > 0);

    // Pad to next power of 2
    const n = std.math.ceilPowerOfTwo(usize, leaves.len) catch unreachable;

    // Count total nodes: sum of all levels = 2*n - 1, but we store level by level
    // Level 0 has n nodes, level 1 has n/2, ..., level log2(n) has 1
    // Total = n + n/2 + ... + 1 = 2*n - 1
    const total_nodes = 2 * n - 1;
    var nodes = try allocator.alloc(Hash, total_nodes);
    errdefer allocator.free(nodes);

    // Level 0: leaves (padded with zero hashes)
    @memcpy(nodes[0..leaves.len], leaves);
    for (leaves.len..n) |i| {
        nodes[i] = zero_hash;
    }

    // Build parent levels
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

    const root = nodes[total_nodes - 1];
    return .{
        .nodes = nodes,
        .n_leaves = leaves.len,
        .padded_size = n,
        .root = root,
        .allocator = allocator,
    };
}

/// Compute Merkle root without storing intermediate nodes. O(log N) stack.
pub fn computeRoot(allocator: std.mem.Allocator, leaves: []const Hash) !Hash {
    std.debug.assert(leaves.len > 0);

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
    std.debug.assert(index < tree.n_leaves);

    const depth = std.math.log2(tree.padded_size);
    var siblings = try allocator.alloc(Hash, depth);
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

// ---------------------------------------------------------------------------
// Domain-separated hashing functions (must match Rust byte-for-byte)
// ---------------------------------------------------------------------------

/// Hash a retained token state for trace commitment (domain: "vi-retained-v3").
pub fn hashRetainedStateDirect(layers: []const RetainedLayerInput) Hash {
    var h = Sha256.init(.{});
    h.update("vi-retained-v3");

    for (layers) |ls| {
        // Hash a (i8 slice as u8 bytes)
        h.update(std.mem.sliceAsBytes(ls.a));
        h.update(std.mem.asBytes(&ls.scale_a));
        if (ls.x_attn_i8) |xa| {
            h.update(&[_]u8{0x01});
            h.update(std.mem.sliceAsBytes(xa));
            h.update(std.mem.asBytes(&ls.scale_x_attn.?));
        } else {
            h.update(&[_]u8{0x00});
        }
    }

    return h.finalResult();
}

/// Input for retained state hashing.
pub const RetainedLayerInput = struct {
    a: []const i8,
    scale_a: f32,
    x_attn_i8: ?[]const i8,
    scale_x_attn: ?f32,
};

/// Compute V4 IO chain hash: H("vi-io-v4" || leaf_hash || token_id LE || prev_io).
pub fn ioHashV4(leaf_hash: Hash, token_id: u32, prev_io_hash: Hash) Hash {
    var h = Sha256.init(.{});
    h.update("vi-io-v4");
    h.update(&leaf_hash);
    h.update(&std.mem.toBytes(token_id));
    h.update(&prev_io_hash);
    return h.finalResult();
}

/// Hash a KV entry: H("vi-kv-v1" || layer LE || position LE || k_roped f64 bytes || v_deq f64 bytes).
pub fn hashKvEntry(layer: u32, position: u32, k_roped: []const f64, v_deq: []const f64) Hash {
    var h = Sha256.init(.{});
    h.update("vi-kv-v1");
    h.update(&std.mem.toBytes(layer));
    h.update(&std.mem.toBytes(position));
    h.update(std.mem.sliceAsBytes(k_roped));
    h.update(std.mem.sliceAsBytes(v_deq));
    return h.finalResult();
}

/// Hash an embedding table row: H("vi-embedding-v1" || row f32 bytes).
pub fn hashEmbeddingRow(row: []const f32) Hash {
    var h = Sha256.init(.{});
    h.update("vi-embedding-v1");
    h.update(std.mem.sliceAsBytes(row));
    return h.finalResult();
}

// ===========================================================================
// Tests
// ===========================================================================

test "merkle_single_leaf" {
    const allocator = std.testing.allocator;
    const leaf = hashLeaf("hello");
    const leaves = [_]Hash{leaf};
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    // Single leaf: root == leaf (padded_size=1, no pairs to hash)
    try std.testing.expectEqualSlices(u8, &leaf, &tree.root);
}

test "merkle_two_leaves" {
    const allocator = std.testing.allocator;
    const leaf0 = hashLeaf("hello");
    const leaf1 = hashLeaf("world");
    const leaves = [_]Hash{ leaf0, leaf1 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    const expected = hashPair(&leaf0, &leaf1);
    try std.testing.expectEqualSlices(u8, &expected, &tree.root);
}

test "merkle_prove_verify_roundtrip" {
    const allocator = std.testing.allocator;
    const leaf0 = hashLeaf("a");
    const leaf1 = hashLeaf("b");
    const leaf2 = hashLeaf("c");
    const leaves = [_]Hash{ leaf0, leaf1, leaf2 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    // Prove and verify each leaf
    for (0..3) |i| {
        var prf = try prove(allocator, &tree, i);
        defer prf.deinit();
        try std.testing.expect(verify(&tree.root, &leaves[i], &prf));
    }
}

test "merkle_wrong_leaf_fails" {
    const allocator = std.testing.allocator;
    const leaf0 = hashLeaf("a");
    const leaf1 = hashLeaf("b");
    const leaves = [_]Hash{ leaf0, leaf1 };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    var prf = try prove(allocator, &tree, 0);
    defer prf.deinit();

    // Verify with wrong leaf should fail
    const wrong = hashLeaf("wrong");
    try std.testing.expect(!verify(&tree.root, &wrong, &prf));
}

test "compute_root_matches_build_tree" {
    const allocator = std.testing.allocator;
    const leaves = [_]Hash{
        hashLeaf("a"),
        hashLeaf("b"),
        hashLeaf("c"),
        hashLeaf("d"),
        hashLeaf("e"),
    };
    var tree = try buildTree(allocator, &leaves);
    defer tree.deinit();

    const root = try computeRoot(allocator, &leaves);
    try std.testing.expectEqualSlices(u8, &tree.root, &root);
}
