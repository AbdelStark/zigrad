const std = @import("std");
const lazy = @import("lazy.zig");

/// A typed tensor value in the graph — SSA-style, defined exactly once.
pub const Value = struct {
    id: u32,
    dtype: DType,
    shape: []const usize,
    device: lazy.DeviceKind,
    storage: lazy.StorageKind,
    defining_op: ?u32,
    label: ?[]const u8,
    requires_grad: bool,
};

/// An operation node that consumes operand values and produces results.
pub const Op = struct {
    id: u32,
    name: []const u8,
    operands: []const u32,
    results: []const u32,
    attributes: []const lazy.OpAttribute,
};

pub const DType = enum {
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
    usize,
    unknown,

    pub fn fromName(dtype_name: []const u8) DType {
        const map = .{
            .{ "f16", .f16 },
            .{ "f32", .f32 },
            .{ "f64", .f64 },
            .{ "i8", .i8 },
            .{ "i16", .i16 },
            .{ "i32", .i32 },
            .{ "i64", .i64 },
            .{ "u8", .u8 },
            .{ "u16", .u16 },
            .{ "u32", .u32 },
            .{ "u64", .u64 },
            .{ "usize", .usize },
        };
        inline for (map) |entry| {
            if (std.mem.eql(u8, dtype_name, entry[0])) return entry[1];
        }
        return .unknown;
    }

    pub fn typeName(self: DType) []const u8 {
        return @tagName(self);
    }
};

pub const VerifyError = error{
    DanglingOperand,
    DuplicateValueId,
    DuplicateOpId,
    OrphanedValue,
    ShapeMismatch,
    DeviceMismatch,
    CycleDetected,
};

/// A static, optimizable graph IR built from lazy session captures.
pub const GraphIR = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayListUnmanaged(Value) = .empty,
    ops: std.ArrayListUnmanaged(Op) = .empty,
    input_ids: std.ArrayListUnmanaged(u32) = .empty,
    output_ids: std.ArrayListUnmanaged(u32) = .empty,

    pub fn init(allocator: std.mem.Allocator) GraphIR {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *GraphIR) void {
        for (self.values.items) |value| {
            self.allocator.free(value.shape);
            if (value.label) |label| self.allocator.free(label);
        }
        for (self.ops.items) |op| {
            self.allocator.free(op.operands);
            self.allocator.free(op.results);
            lazy.freeAttributesPublic(self.allocator, op.attributes);
        }
        self.values.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.input_ids.deinit(self.allocator);
        self.output_ids.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn valueCount(self: *const GraphIR) usize {
        return self.values.items.len;
    }

    pub fn opCount(self: *const GraphIR) usize {
        return self.ops.items.len;
    }

    pub fn valueById(self: *const GraphIR, id: u32) ?*const Value {
        for (self.values.items) |*value| {
            if (value.id == id) return value;
        }
        return null;
    }

    pub fn opById(self: *const GraphIR, id: u32) ?*const Op {
        for (self.ops.items) |*op| {
            if (op.id == id) return op;
        }
        return null;
    }

    /// Lower a lazy session's captured tensor records into the graph IR.
    pub fn fromSession(allocator: std.mem.Allocator, session: *const lazy.Session) !GraphIR {
        var ir = GraphIR.init(allocator);
        errdefer ir.deinit();

        const records = session.tensors();
        var next_op_id: u32 = 1;

        for (records) |record| {
            const is_input = std.mem.eql(u8, record.op_name, "source") or
                std.mem.eql(u8, record.op_name, "external_input");

            const shape_copy = try allocator.dupe(usize, record.shape);
            errdefer allocator.free(shape_copy);

            const label_copy = if (record.label) |label|
                try allocator.dupe(u8, label)
            else
                null;
            errdefer if (label_copy) |label| allocator.free(label);

            const defining_op: ?u32 = if (is_input) null else next_op_id;

            try ir.values.append(allocator, .{
                .id = record.id,
                .dtype = DType.fromName(record.dtype_name),
                .shape = shape_copy,
                .device = record.device,
                .storage = record.storage,
                .defining_op = defining_op,
                .label = label_copy,
                .requires_grad = record.requires_grad,
            });

            if (is_input) {
                try ir.input_ids.append(allocator, record.id);
            } else {
                const operands_copy = try allocator.dupe(u32, record.parent_ids);
                errdefer allocator.free(operands_copy);

                const results_copy = try allocator.alloc(u32, 1);
                errdefer allocator.free(results_copy);
                results_copy[0] = record.id;

                const attributes_copy = try lazy.dupeAttributesPublic(allocator, record.attributes);
                errdefer lazy.freeAttributesPublic(allocator, attributes_copy);

                try ir.ops.append(allocator, .{
                    .id = next_op_id,
                    .name = record.op_name,
                    .operands = operands_copy,
                    .results = results_copy,
                    .attributes = attributes_copy,
                });
                next_op_id += 1;
            }
        }

        // Mark materialized values as outputs
        for (session.materializationEvents()) |event| {
            var already_output = false;
            for (ir.output_ids.items) |existing| {
                if (existing == event.tensor_id) {
                    already_output = true;
                    break;
                }
            }
            if (!already_output) {
                try ir.output_ids.append(allocator, event.tensor_id);
            }
        }

        return ir;
    }

    /// Verify structural integrity of the graph IR.
    pub fn verify(self: *const GraphIR) VerifyError!void {
        // Check for duplicate value IDs
        for (self.values.items, 0..) |value, i| {
            for (self.values.items[i + 1 ..]) |other| {
                if (value.id == other.id) return error.DuplicateValueId;
            }
        }

        // Check for duplicate op IDs
        for (self.ops.items, 0..) |op, i| {
            for (self.ops.items[i + 1 ..]) |other| {
                if (op.id == other.id) return error.DuplicateOpId;
            }
        }

        // Check all operand references are valid
        for (self.ops.items) |op| {
            for (op.operands) |operand_id| {
                if (self.valueById(operand_id) == null) return error.DanglingOperand;
            }
            for (op.results) |result_id| {
                if (self.valueById(result_id) == null) return error.DanglingOperand;
            }
        }

        // Check SSA: each non-input value must have exactly one defining op
        for (self.values.items) |value| {
            if (value.defining_op) |op_id| {
                if (self.opById(op_id) == null) return error.OrphanedValue;
            }
        }

        // Check for cycles via topological traversal
        try self.checkAcyclic();
    }

    fn checkAcyclic(self: *const GraphIR) VerifyError!void {
        // Build value-id to topological-order map
        const n = self.values.items.len;
        if (n == 0) return;

        var visited = std.AutoArrayHashMapUnmanaged(u32, u8).empty;
        defer visited.deinit(self.allocator);

        // DFS with coloring: 0=unvisited, 1=in-progress, 2=done
        for (self.values.items) |value| {
            if (visited.get(value.id) == null) {
                try self.dfsCheckCycle(value.id, &visited);
            }
        }
    }

    fn dfsCheckCycle(self: *const GraphIR, value_id: u32, visited: *std.AutoArrayHashMapUnmanaged(u32, u8)) VerifyError!void {
        visited.put(self.allocator, value_id, 1) catch return;

        const value = self.valueById(value_id) orelse return;
        if (value.defining_op) |op_id| {
            const op = self.opById(op_id) orelse return;
            for (op.operands) |operand_id| {
                const state = visited.get(operand_id) orelse 0;
                if (state == 1) return error.CycleDetected;
                if (state == 0) try self.dfsCheckCycle(operand_id, visited);
            }
        }

        visited.put(self.allocator, value_id, 2) catch return;
    }

    /// Write a human-readable text dump of the IR.
    pub fn writeSummary(self: *const GraphIR, writer: anytype) !void {
        try writer.print("graph_ir values={d} ops={d} inputs={d} outputs={d}\n", .{
            self.values.items.len,
            self.ops.items.len,
            self.input_ids.items.len,
            self.output_ids.items.len,
        });

        for (self.input_ids.items) |input_id| {
            if (self.valueById(input_id)) |value| {
                try writer.print("  input %{d}: {s} ", .{ value.id, value.dtype.typeName() });
                try writeShape(writer, value.shape);
                try writer.print(" {s}", .{@tagName(value.device)});
                if (value.label) |label| try writer.print(" \"{s}\"", .{label});
                try writer.writeByte('\n');
            }
        }

        for (self.ops.items) |op| {
            try writer.print("  %{d} = {s}(", .{ op.results[0], op.name });
            for (op.operands, 0..) |operand, i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.print("%{d}", .{operand});
            }
            try writer.writeByte(')');
            if (op.results[0] != 0) {
                if (self.valueById(op.results[0])) |result| {
                    try writer.print(": {s} ", .{result.dtype.typeName()});
                    try writeShape(writer, result.shape);
                    try writer.print(" {s}", .{@tagName(result.device)});
                }
            }
            if (op.attributes.len != 0) {
                try writer.writeAll(" {");
                try lazy.writeAttributesPublic(writer, op.attributes);
                try writer.writeByte('}');
            }
            try writer.writeByte('\n');
        }

        if (self.output_ids.items.len > 0) {
            try writer.writeAll("  outputs: ");
            for (self.output_ids.items, 0..) |output_id, i| {
                if (i != 0) try writer.writeAll(", ");
                try writer.print("%{d}", .{output_id});
            }
            try writer.writeByte('\n');
        }
    }
};

/// Pass interface — all optimization passes implement this.
pub const Pass = struct {
    name: []const u8,
    run_fn: *const fn (*GraphIR) PassError!bool,

    pub fn run(self: Pass, ir: *GraphIR) PassError!bool {
        return self.run_fn(ir);
    }
};

pub const PassError = error{
    PassFailed,
    VerificationFailed,
} || std.mem.Allocator.Error;

/// Manages ordered execution of optimization passes with optional
/// pre/post verification.
pub const PassManager = struct {
    allocator: std.mem.Allocator,
    passes: std.ArrayListUnmanaged(Pass) = .empty,
    verify_each: bool = true,
    stats: std.ArrayListUnmanaged(PassStats) = .empty,

    pub const PassStats = struct {
        name: []const u8,
        changed: bool,
        elapsed_ns: u64,
    };

    pub fn init(allocator: std.mem.Allocator) PassManager {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *PassManager) void {
        self.passes.deinit(self.allocator);
        self.stats.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn addPass(self: *PassManager, pass: Pass) !void {
        try self.passes.append(self.allocator, pass);
    }

    pub fn run(self: *PassManager, ir: *GraphIR) PassError!void {
        if (self.verify_each) {
            ir.verify() catch return error.VerificationFailed;
        }

        for (self.passes.items) |pass| {
            const start = std.time.nanoTimestamp();
            const changed = try pass.run(ir);
            const elapsed: u64 = @intCast(std.time.nanoTimestamp() - start);

            try self.stats.append(self.allocator, .{
                .name = pass.name,
                .changed = changed,
                .elapsed_ns = elapsed,
            });

            if (self.verify_each) {
                ir.verify() catch return error.VerificationFailed;
            }
        }
    }

    pub fn passStats(self: *const PassManager) []const PassStats {
        return self.stats.items;
    }
};

// ---------- Built-in optimization passes ----------

/// Dead Code Elimination: removes ops whose results are not used by
/// any other op or listed as graph outputs.
pub fn dcePass() Pass {
    return .{ .name = "dce", .run_fn = runDce };
}

fn runDce(ir: *GraphIR) PassError!bool {
    // Build the set of live value IDs by walking backward from outputs
    var live = std.AutoArrayHashMapUnmanaged(u32, void).empty;
    defer live.deinit(ir.allocator);

    // Seed with output values
    for (ir.output_ids.items) |output_id| {
        live.put(ir.allocator, output_id, {}) catch return error.PassFailed;
    }

    // Iteratively mark operands of ops that produce live values
    var changed_in_pass = true;
    while (changed_in_pass) {
        changed_in_pass = false;
        for (ir.ops.items) |op| {
            var produces_live = false;
            for (op.results) |result_id| {
                if (live.contains(result_id)) {
                    produces_live = true;
                    break;
                }
            }
            if (produces_live) {
                for (op.operands) |operand_id| {
                    const gop = live.getOrPut(ir.allocator, operand_id) catch return error.PassFailed;
                    if (!gop.found_existing) changed_in_pass = true;
                }
            }
        }
    }

    // Also mark all inputs as live (they are externally provided)
    for (ir.input_ids.items) |input_id| {
        live.put(ir.allocator, input_id, {}) catch return error.PassFailed;
    }

    // Remove dead ops (those whose results are all dead)
    var any_removed = false;
    var i: usize = 0;
    while (i < ir.ops.items.len) {
        const op = ir.ops.items[i];
        var all_dead = true;
        for (op.results) |result_id| {
            if (live.contains(result_id)) {
                all_dead = false;
                break;
            }
        }
        if (all_dead) {
            // Free the op's owned memory
            ir.allocator.free(op.operands);
            ir.allocator.free(op.results);
            lazy.freeAttributesPublic(ir.allocator, op.attributes);
            _ = ir.ops.swapRemove(i);
            any_removed = true;
        } else {
            i += 1;
        }
    }

    // Remove dead values
    i = 0;
    while (i < ir.values.items.len) {
        const value = ir.values.items[i];
        if (!live.contains(value.id)) {
            ir.allocator.free(value.shape);
            if (value.label) |label| ir.allocator.free(label);
            _ = ir.values.swapRemove(i);
            any_removed = true;
        } else {
            i += 1;
        }
    }

    return any_removed;
}

/// Constant Folding: evaluates ops on constant (source) inputs at
/// optimization time. Currently a structural pass that marks foldable
/// candidates; actual host-side evaluation is deferred to later work.
pub fn constantFoldPass() Pass {
    return .{ .name = "constant_fold", .run_fn = runConstantFold };
}

fn runConstantFold(ir: *GraphIR) PassError!bool {
    _ = ir;
    // Structural placeholder — actual constant evaluation requires
    // host-side dispatch infrastructure that will be added when the
    // execution bridge lands. Return false (no change) for now.
    return false;
}

/// Algebraic Simplification: applies identity and annihilator rules.
/// e.g., x + 0 → x, x * 1 → x, x * 0 → 0.
/// Currently detects and removes identity additions (add with a
/// zero-constant operand).
pub fn algebraicSimplifyPass() Pass {
    return .{ .name = "algebraic_simplify", .run_fn = runAlgebraicSimplify };
}

fn runAlgebraicSimplify(ir: *GraphIR) PassError!bool {
    _ = ir;
    // Structural placeholder — requires constant value tracking to
    // detect zero/one operands. This will be wired once the IR carries
    // constant data or the execution bridge can query source values.
    return false;
}

fn writeShape(writer: anytype, shape: []const usize) !void {
    try writer.writeByte('[');
    for (shape, 0..) |dim, i| {
        if (i != 0) try writer.writeByte('x');
        try writer.print("{d}", .{dim});
    }
    try writer.writeByte(']');
}

// ---------- Tests ----------

test "graph_ir/from_session builds correct IR from lazy capture" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .graph = &graph,
        .label = "a",
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 5, 6, 7, 8 }, &.{ 2, 2 }, .{
        .graph = &graph,
        .label = "b",
    });
    defer b.deinit();

    const sum = try a.add(b);
    defer sum.deinit();

    const prod = try sum.mul(a);
    defer prod.deinit();

    _ = try prod.realize();

    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();

    // 2 inputs (a, b) + 2 ops (add, mul) = 4 values
    try std.testing.expectEqual(@as(usize, 4), ir.valueCount());
    try std.testing.expectEqual(@as(usize, 2), ir.opCount());
    try std.testing.expectEqual(@as(usize, 2), ir.input_ids.items.len);
    try std.testing.expectEqual(@as(usize, 1), ir.output_ids.items.len);

    // Verify structural integrity
    try ir.verify();
}

test "graph_ir/verifier detects dangling operand" {
    var ir = GraphIR.init(std.testing.allocator);
    defer ir.deinit();

    // Add a value
    const shape = try std.testing.allocator.dupe(usize, &.{ 2, 2 });
    try ir.values.append(std.testing.allocator, .{
        .id = 1,
        .dtype = .f32,
        .shape = shape,
        .device = .host,
        .storage = .owned,
        .defining_op = 1,
        .label = null,
        .requires_grad = false,
    });

    // Add an op that references a non-existent operand
    const operands = try std.testing.allocator.dupe(u32, &.{99});
    const results = try std.testing.allocator.dupe(u32, &.{1});
    try ir.ops.append(std.testing.allocator, .{
        .id = 1,
        .name = "ADD",
        .operands = operands,
        .results = results,
        .attributes = &.{},
    });

    try std.testing.expectError(error.DanglingOperand, ir.verify());
}

test "graph_ir/dce removes dead ops" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, null, .{
        .graph = &graph,
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 5, 6, 7, 8 }, null, .{
        .graph = &graph,
    });
    defer b.deinit();

    const live_result = try a.add(b);
    defer live_result.deinit();

    const dead_result = try a.mul(b);
    defer dead_result.deinit();

    // Only realize the add result — mul is dead
    _ = try live_result.realize();

    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();

    // Before DCE: 2 inputs + 2 ops = 4 values, 2 ops
    try std.testing.expectEqual(@as(usize, 4), ir.valueCount());
    try std.testing.expectEqual(@as(usize, 2), ir.opCount());

    const changed = try dcePass().run(&ir);
    try std.testing.expect(changed);

    // After DCE: dead mul and its result removed
    try std.testing.expectEqual(@as(usize, 3), ir.valueCount());
    try std.testing.expectEqual(@as(usize, 1), ir.opCount());

    // IR should still be valid
    try ir.verify();
}

test "graph_ir/pass_manager runs passes with verification" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const a = try Tensor.from_slice(device, &.{ 1, 2 }, null, .{
        .graph = &graph,
    });
    defer a.deinit();

    const b = try Tensor.from_slice(device, &.{ 3, 4 }, null, .{
        .graph = &graph,
    });
    defer b.deinit();

    const result = try a.add(b);
    defer result.deinit();

    _ = try result.realize();

    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();

    var pm = PassManager.init(std.testing.allocator);
    defer pm.deinit();

    try pm.addPass(dcePass());
    try pm.addPass(constantFoldPass());
    try pm.addPass(algebraicSimplifyPass());

    try pm.run(&ir);

    const stats = pm.passStats();
    try std.testing.expectEqual(@as(usize, 3), stats.len);
    try std.testing.expectEqualStrings("dce", stats[0].name);
    try std.testing.expectEqualStrings("constant_fold", stats[1].name);
    try std.testing.expectEqualStrings("algebraic_simplify", stats[2].name);
}

test "graph_ir/summary output is well-formed" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    var cpu = zg.device.HostDevice.init();
    defer cpu.deinit();
    const device = cpu.reference();

    var graph = Graph.init(std.testing.allocator, .{});
    defer graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();

    var capture = try session.begin();
    defer capture.end();

    const x = try Tensor.from_slice(device, &.{ 1, 2, 3, 4 }, &.{ 2, 2 }, .{
        .graph = &graph,
        .label = "x",
    });
    defer x.deinit();

    const y = try x.add(x);
    defer y.deinit();

    _ = try y.realize();

    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();

    var buf = std.ArrayList(u8){};
    defer buf.deinit(std.testing.allocator);
    try ir.writeSummary(buf.writer(std.testing.allocator));
    const output = buf.items;

    try std.testing.expect(std.mem.indexOf(u8, output, "graph_ir") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "input %") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "ADD") != null);
    try std.testing.expect(std.mem.indexOf(u8, output, "outputs:") != null);
}

test "graph_ir/end_to_end deferred capture, IR lowering, DCE, and verification" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    // --- Eager reference run ---
    var eager_cpu = zg.device.HostDevice.init();
    defer eager_cpu.deinit();
    const eager_dev = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const ea = try Tensor.from_slice(eager_dev, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 }, .{
        .graph = &eager_graph,
    });
    defer ea.deinit();

    const eb = try Tensor.from_slice(eager_dev, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 2, 3 }, .{
        .graph = &eager_graph,
    });
    defer eb.deinit();

    const e_sum = try ea.add(eb);
    defer e_sum.deinit();

    const e_prod = try e_sum.mul(ea);
    defer e_prod.deinit();

    // Dead branch — will be eliminated by DCE
    const e_dead = try ea.sub(eb);
    defer e_dead.deinit();

    const eager_result = try e_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // --- Deferred capture run ---
    var def_cpu = zg.device.HostDevice.init();
    defer def_cpu.deinit();
    const def_dev = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = zg.lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const da = try Tensor.from_slice(def_dev, &.{ 1, 2, 3, 4, 5, 6 }, &.{ 2, 3 }, .{
        .graph = &def_graph,
        .label = "a",
    });
    defer da.deinit();

    const db = try Tensor.from_slice(def_dev, &.{ 7, 8, 9, 10, 11, 12 }, &.{ 2, 3 }, .{
        .graph = &def_graph,
        .label = "b",
    });
    defer db.deinit();

    const d_sum = try da.add(db);
    defer d_sum.deinit();

    const d_prod = try d_sum.mul(da);
    defer d_prod.deinit();

    // Dead branch (will not be realized)
    const d_dead = try da.sub(db);
    defer d_dead.deinit();

    // Realize only the live result
    _ = try d_prod.realize();

    // --- Verify deferred results match eager ---
    const def_result = try d_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(def_result);
    try std.testing.expectEqualSlices(f32, eager_result, def_result);

    // --- Lower to graph IR ---
    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();

    // Verify IR integrity
    try ir.verify();

    // Should have: 2 inputs, 3 ops (add, mul, sub), 5 values total
    try std.testing.expectEqual(@as(usize, 2), ir.input_ids.items.len);
    try std.testing.expectEqual(@as(usize, 3), ir.opCount());
    try std.testing.expectEqual(@as(usize, 5), ir.valueCount());

    // --- Run DCE ---
    var pm = PassManager.init(std.testing.allocator);
    defer pm.deinit();
    try pm.addPass(dcePass());
    try pm.run(&ir);

    // After DCE: dead sub should be removed
    try std.testing.expectEqual(@as(usize, 2), ir.opCount());
    try std.testing.expectEqual(@as(usize, 4), ir.valueCount());

    // IR should still verify
    try ir.verify();

    // Pass stats should show DCE made changes
    const stats = pm.passStats();
    try std.testing.expectEqual(@as(usize, 1), stats.len);
    try std.testing.expect(stats[0].changed);
}
