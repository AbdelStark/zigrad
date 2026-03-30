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

    /// Return ops in topological (dependency) order. Caller owns the returned slice.
    pub fn topoSort(self: *const GraphIR, allocator: std.mem.Allocator) ![]u32 {
        const ops = self.ops.items;
        const n = ops.len;
        if (n == 0) return allocator.alloc(u32, 0);

        // Build: result_value_id -> op_index (which op produces this value)
        var producer = std.AutoArrayHashMapUnmanaged(u32, usize).empty;
        defer producer.deinit(allocator);
        for (ops, 0..) |op, idx| {
            for (op.results) |result_id| {
                try producer.put(allocator, result_id, idx);
            }
        }

        // Compute in-degree per op (number of operand-producing ops)
        const in_degree = try allocator.alloc(usize, n);
        defer allocator.free(in_degree);
        @memset(in_degree, 0);

        for (ops, 0..) |op, idx| {
            for (op.operands) |operand_id| {
                if (producer.get(operand_id)) |_| {
                    in_degree[idx] += 1;
                }
            }
        }

        // Initialize queue with zero-degree ops
        var queue = std.ArrayListUnmanaged(usize).empty;
        defer queue.deinit(allocator);
        for (0..n) |idx| {
            if (in_degree[idx] == 0) try queue.append(allocator, idx);
        }

        var sorted = try std.ArrayListUnmanaged(u32).initCapacity(allocator, n);
        errdefer sorted.deinit(allocator);

        var processed: usize = 0;
        while (processed < queue.items.len) {
            const op_idx = queue.items[processed];
            processed += 1;
            sorted.appendAssumeCapacity(ops[op_idx].id);

            // For each value this op produces, find downstream consumers
            for (ops[op_idx].results) |result_id| {
                for (ops, 0..) |other_op, other_idx| {
                    for (other_op.operands) |operand_id| {
                        if (operand_id == result_id) {
                            in_degree[other_idx] -= 1;
                            if (in_degree[other_idx] == 0) {
                                try queue.append(allocator, other_idx);
                            }
                        }
                    }
                }
            }
        }

        if (sorted.items.len != n) return error.CycleDetected;
        return sorted.toOwnedSlice(allocator);
    }

    /// Execute the graph on a device, producing output buffers.
    ///
    /// `T` is the element type (f32, f64). All values in the graph must
    /// match this type.
    ///
    /// `inputs` maps graph input value IDs to their host-resident data
    /// slices. The execution bridge copies input data to device buffers.
    ///
    /// Returns an `ExecutionResult(T)` whose `.getOutput(id)` method
    /// retrieves computed data for any graph output value ID.
    pub fn execute(
        self: *const GraphIR,
        comptime T: type,
        allocator: std.mem.Allocator,
        device: anytype,
        inputs: std.AutoArrayHashMapUnmanaged(u32, []const T),
    ) ExecuteError!ExecutionResult(T) {
        const DevRef = @import("device/device_reference.zig");
        const dev: DevRef = device;

        // Suspend any active lazy session so dispatch calls execute
        // immediately rather than being queued as deferred thunks.
        const saved_session = lazy.suspendSession();
        defer lazy.restoreSession(saved_session);

        // Validate dtype
        const expected_dtype = DType.fromName(@typeName(T));
        for (self.values.items) |value| {
            if (value.dtype != expected_dtype and value.dtype != .unknown) {
                return error.UnsupportedDType;
            }
        }

        // Topological sort
        const sorted_op_ids = self.topoSort(allocator) catch |e| switch (e) {
            error.CycleDetected => return error.CycleDetected,
            else => return error.OutOfMemory,
        };
        defer allocator.free(sorted_op_ids);

        var result = ExecutionResult(T){
            .buffers = .empty,
            .owned_buffers = .empty,
            .allocator = allocator,
            .device = dev,
        };
        errdefer result.deinit();

        // Bind input buffers (copy to device)
        for (self.input_ids.items) |input_id| {
            const src = inputs.get(input_id) orelse return error.MissingInput;
            const buf = dev.mem_alloc(T, src.len) catch return error.OutOfMemory;
            dev.mem_copy(T, src, buf);
            result.owned_buffers.append(allocator, buf) catch return error.OutOfMemory;
            result.buffers.put(allocator, input_id, buf) catch return error.OutOfMemory;
        }

        // Execute ops in topological order
        for (sorted_op_ids) |op_id| {
            const op = self.opById(op_id) orelse return error.MissingValue;
            try executeOp(T, op, self, &result, allocator, dev);
        }

        return result;
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

fn shapeSize(shape: []const usize) usize {
    var size: usize = 1;
    for (shape) |d| size *= d;
    return size;
}

// ---------- Execution Bridge ----------

pub const ExecuteError = error{
    UnsupportedDType,
    UnsupportedOp,
    MissingInput,
    MissingOperand,
    MissingValue,
    CycleDetected,
    OutOfMemory,
};

/// Typed execution result — owns all intermediate and output buffers.
pub fn ExecutionResult(comptime T: type) type {
    const DevRef = @import("device/device_reference.zig");
    return struct {
        const Self = @This();
        /// Maps value IDs to their computed data slices.
        buffers: std.AutoArrayHashMapUnmanaged(u32, []T),
        /// Tracks buffers that were allocated by the execution bridge
        /// (not borrowed from inputs) so they can be freed.
        owned_buffers: std.ArrayListUnmanaged([]T),
        allocator: std.mem.Allocator,
        device: DevRef,

        pub fn deinit(self: *Self) void {
            for (self.owned_buffers.items) |buf| {
                self.device.mem_free(buf);
            }
            self.owned_buffers.deinit(self.allocator);
            self.buffers.deinit(self.allocator);
        }

        /// Retrieve the output buffer for a given value ID.
        pub fn getOutput(self: *const Self, id: u32) ?[]const T {
            return self.buffers.get(id);
        }
    };
}

/// Input binding for execute(): maps a graph input value ID to data.


fn executeOp(
    comptime T: type,
    op: *const Op,
    ir: *const GraphIR,
    result: *ExecutionResult(T),
    allocator: std.mem.Allocator,
    device: anytype,
) ExecuteError!void {
    const opspec = @import("device/opspec.zig");
    const result_id = op.results[0];
    const result_value = ir.valueById(result_id) orelse return error.MissingValue;
    const result_size = shapeSize(result_value.shape);

    // Helper to look up an operand buffer
    const getOperand = struct {
        fn get(buffers: *const std.AutoArrayHashMapUnmanaged(u32, []T), id: u32) ExecuteError![]T {
            return buffers.get(id) orelse return error.MissingOperand;
        }
    }.get;

    // Helper to allocate an output buffer
    const allocOut = struct {
        fn alloc(dev: anytype, res: *ExecutionResult(T), alloc_: std.mem.Allocator, n: usize) ExecuteError![]T {
            const buf = dev.mem_alloc(T, n) catch return error.OutOfMemory;
            res.owned_buffers.append(alloc_, buf) catch return error.OutOfMemory;
            return buf;
        }
    }.alloc;

    const name = op.name;

    // ----- Elementwise binary ops -----
    if (std.mem.eql(u8, name, "ADD")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try getOperand(&result.buffers, op.operands[1]);
        const z = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.add(T){ .x = x, .y = y, .z = z });
        result.buffers.put(allocator, result_id, z) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "SUB")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try getOperand(&result.buffers, op.operands[1]);
        const z = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.sub(T){ .x = x, .y = y, .z = z });
        result.buffers.put(allocator, result_id, z) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "MUL")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try getOperand(&result.buffers, op.operands[1]);
        const z = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.mul(T){ .x = x, .y = y, .z = z });
        result.buffers.put(allocator, result_id, z) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "DIV")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try getOperand(&result.buffers, op.operands[1]);
        const z = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.div(T){ .x = x, .y = y, .z = z });
        result.buffers.put(allocator, result_id, z) catch return error.OutOfMemory;

        // ----- Unary ops -----
    } else if (std.mem.eql(u8, name, "EXP")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.exp_fwd(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "sqrt")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.sqrt_fwd(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "rsqrt")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.rsqrt_fwd(T){ .x = x, .y = y, .eps = 1e-7 });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "relu")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.relu_fwd(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "tanh")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        device.dispatch(opspec.tanh_fwd(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "POW")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, result_size);
        const exp_val = getAttrFloat(op.attributes, "exponent") orelse 2.0;
        device.dispatch(opspec.pow_fwd(T){ .x = x, .exp = @floatCast(exp_val), .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;

        // ----- Reductions -----
    } else if (std.mem.eql(u8, name, "SUM")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, 1);
        device.dispatch(opspec.sum(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;
    } else if (std.mem.eql(u8, name, "MAX")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try allocOut(device, result, allocator, 1);
        device.dispatch(opspec.max_fwd(T){ .x = x, .y = y });
        result.buffers.put(allocator, result_id, y) catch return error.OutOfMemory;

        // ----- Transpose -----
    } else if (std.mem.eql(u8, name, "TRANSPOSE")) {
        const a = try getOperand(&result.buffers, op.operands[0]);
        const b = try allocOut(device, result, allocator, result_size);
        const src_value = ir.valueById(op.operands[0]) orelse return error.MissingValue;
        const m = if (src_value.shape.len >= 2) src_value.shape[0] else 1;
        const n = if (src_value.shape.len >= 2) src_value.shape[1] else src_value.shape[0];
        device.dispatch(opspec.transpose(T){ .A = a, .B = b, .m = m, .n = n, .alpha = 0.0 });
        result.buffers.put(allocator, result_id, b) catch return error.OutOfMemory;

        // ----- Matrix multiply -----
    } else if (std.mem.eql(u8, name, "MATMUL_AB") or
        std.mem.eql(u8, name, "MATMUL_AtB") or
        std.mem.eql(u8, name, "MATMUL_ABt") or
        std.mem.eql(u8, name, "MATMUL_AtBt"))
    {
        try executeMatmul(T, op, ir, result, allocator, device);
    } else if (std.mem.eql(u8, name, "DOT")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const y = try getOperand(&result.buffers, op.operands[1]);
        const z = try allocOut(device, result, allocator, 1);
        device.dispatch(opspec.dot(T){ .x = x, .y = y, .z = z });
        result.buffers.put(allocator, result_id, z) catch return error.OutOfMemory;

        // ----- Pass-through / view ops -----
    } else if (std.mem.eql(u8, name, "clone") or std.mem.eql(u8, name, "alias")) {
        const x = try getOperand(&result.buffers, op.operands[0]);
        const buf = try allocOut(device, result, allocator, result_size);
        device.mem_copy(T, x, buf);
        result.buffers.put(allocator, result_id, buf) catch return error.OutOfMemory;
    } else {
        return error.UnsupportedOp;
    }
}

fn executeMatmul(
    comptime T: type,
    op: *const Op,
    ir: *const GraphIR,
    result: *ExecutionResult(T),
    allocator: std.mem.Allocator,
    device: anytype,
) ExecuteError!void {
    const opspec = @import("device/opspec.zig");
    const result_id = op.results[0];
    const result_value = ir.valueById(result_id) orelse return error.MissingValue;
    const result_size = shapeSize(result_value.shape);

    const a_buf = result.buffers.get(op.operands[0]) orelse return error.MissingOperand;
    const b_buf = result.buffers.get(op.operands[1]) orelse return error.MissingOperand;
    const a_value = ir.valueById(op.operands[0]) orelse return error.MissingValue;
    const b_value = ir.valueById(op.operands[1]) orelse return error.MissingValue;

    const trans_a = getAttrBool(op.attributes, "trans_a") orelse
        (std.mem.eql(u8, op.name, "MATMUL_AtB") or std.mem.eql(u8, op.name, "MATMUL_AtBt"));
    const trans_b = getAttrBool(op.attributes, "trans_b") orelse
        (std.mem.eql(u8, op.name, "MATMUL_ABt") or std.mem.eql(u8, op.name, "MATMUL_AtBt"));
    const alpha: T = if (getAttrFloat(op.attributes, "alpha")) |v| @floatCast(v) else 1.0;
    const beta: T = if (getAttrFloat(op.attributes, "beta")) |v| @floatCast(v) else 0.0;

    // Derive matrix dimensions from shapes
    const a_shape = a_value.shape;
    const b_shape = b_value.shape;

    const a_rows = if (a_shape.len >= 2) a_shape[a_shape.len - 2] else 1;
    const a_cols = if (a_shape.len >= 2) a_shape[a_shape.len - 1] else a_shape[0];
    const b_rows = if (b_shape.len >= 2) b_shape[b_shape.len - 2] else 1;
    const b_cols = if (b_shape.len >= 2) b_shape[b_shape.len - 1] else b_shape[0];

    const M = if (trans_a) a_cols else a_rows;
    const K = if (trans_a) a_rows else a_cols;
    const N = if (trans_b) b_rows else b_cols;

    const c_buf = device.mem_alloc(T, result_size) catch return error.OutOfMemory;
    result.owned_buffers.append(allocator, c_buf) catch return error.OutOfMemory;

    // Zero the output if beta is 0 (matmul may expect initialized memory)
    if (beta == 0) device.mem_fill(T, c_buf, 0);

    // Check if this is a batched matmul (>2D shapes)
    if (a_shape.len > 2 or b_shape.len > 2) {
        // Use bmm_acc opspec for batched matmul
        device.dispatch(opspec.bmm_acc(T){
            .A = a_buf,
            .B = b_buf,
            .C = c_buf,
            .A_shape = a_shape,
            .B_shape = b_shape,
            .C_shape = result_value.shape,
            .trans_a = trans_a,
            .trans_b = trans_b,
            .lda = a_cols,
            .ldb = b_cols,
            .ldc = N,
            .alpha = alpha,
            .beta = beta,
        });
    } else {
        // 2D matmul
        device.dispatch(opspec.matmul(T){
            .A = a_buf,
            .B = b_buf,
            .C = c_buf,
            .m = M,
            .n = N,
            .k = K,
            .trans_a = trans_a,
            .trans_b = trans_b,
            .lda = a_cols,
            .ldb = b_cols,
            .ldc = N,
            .alpha = alpha,
            .beta = beta,
        });
    }

    result.buffers.put(allocator, result_id, c_buf) catch return error.OutOfMemory;
}

fn getAttrBool(attributes: []const lazy.OpAttribute, key: []const u8) ?bool {
    for (attributes) |attr| {
        if (std.mem.eql(u8, attr.key, key)) {
            switch (attr.value) {
                .bool => |v| return v,
                else => return null,
            }
        }
    }
    return null;
}

fn getAttrFloat(attributes: []const lazy.OpAttribute, key: []const u8) ?f64 {
    for (attributes) |attr| {
        if (std.mem.eql(u8, attr.key, key)) {
            switch (attr.value) {
                .float => |v| return v,
                else => return null,
            }
        }
    }
    return null;
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

// ---------- Execution Bridge Tests ----------

test "graph_ir/execute roundtrip parity: elementwise chain" {
    // Eager: a + b, then result * a => should match IR-executed result
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    // --- Eager reference ---
    var eager_cpu = zg.device.HostDevice.init();
    defer eager_cpu.deinit();
    const eager_dev = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const ea = try Tensor.from_slice(eager_dev, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, &.{ 2, 3 }, .{
        .graph = &eager_graph,
    });
    defer ea.deinit();
    const eb = try Tensor.from_slice(eager_dev, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, &.{ 2, 3 }, .{
        .graph = &eager_graph,
    });
    defer eb.deinit();

    const e_sum = try ea.add(eb);
    defer e_sum.deinit();
    const e_prod = try e_sum.mul(ea);
    defer e_prod.deinit();

    const eager_result = try e_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // --- Deferred capture ---
    var def_cpu = zg.device.HostDevice.init();
    defer def_cpu.deinit();
    const def_dev = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const da = try Tensor.from_slice(def_dev, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, &.{ 2, 3 }, .{
        .graph = &def_graph,
        .label = "a",
    });
    defer da.deinit();
    const db = try Tensor.from_slice(def_dev, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }, &.{ 2, 3 }, .{
        .graph = &def_graph,
        .label = "b",
    });
    defer db.deinit();

    const d_sum = try da.add(db);
    defer d_sum.deinit();
    const d_prod = try d_sum.mul(da);
    defer d_prod.deinit();
    _ = try d_prod.realize();

    // --- Lower to IR ---
    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();
    try ir.verify();

    // --- Execute IR ---
    var exec_cpu = zg.device.HostDevice.init();
    defer exec_cpu.deinit();
    const exec_dev = exec_cpu.reference();

    var input_map = std.AutoArrayHashMapUnmanaged(u32, []const f32).empty;
    defer input_map.deinit(std.testing.allocator);
    for (ir.input_ids.items) |input_id| {
        const value = ir.valueById(input_id).?;
        if (std.mem.eql(u8, value.label.?, "a")) {
            try input_map.put(std.testing.allocator, input_id, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
        } else {
            try input_map.put(std.testing.allocator, input_id, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
        }
    }

    var exec_result = try ir.execute(f32, std.testing.allocator, exec_dev, input_map);
    defer exec_result.deinit();

    // Compare outputs
    for (ir.output_ids.items) |output_id| {
        const ir_output = exec_result.getOutput(output_id).?;
        try std.testing.expectEqualSlices(f32, eager_result, ir_output);
    }
}

test "graph_ir/execute roundtrip parity: matmul 2D" {
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    // --- Eager: A @ B where A=[2,3], B=[3,2] => C=[2,2] ---
    var eager_cpu = zg.device.HostDevice.init();
    defer eager_cpu.deinit();
    const eager_dev = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    const ea = try Tensor.from_slice(eager_dev, &a_data, &.{ 2, 3 }, .{
        .graph = &eager_graph,
    });
    defer ea.deinit();
    const eb = try Tensor.from_slice(eager_dev, &b_data, &.{ 3, 2 }, .{
        .graph = &eager_graph,
    });
    defer eb.deinit();

    const e_mm = try ea.bmm(eb, .{});
    defer e_mm.deinit();

    const eager_result = try e_mm.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // --- Deferred capture ---
    var def_cpu = zg.device.HostDevice.init();
    defer def_cpu.deinit();
    const def_dev = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var capture = try session.begin();
    defer capture.end();

    const da = try Tensor.from_slice(def_dev, &a_data, &.{ 2, 3 }, .{
        .graph = &def_graph,
        .label = "a",
    });
    defer da.deinit();
    const db = try Tensor.from_slice(def_dev, &b_data, &.{ 3, 2 }, .{
        .graph = &def_graph,
        .label = "b",
    });
    defer db.deinit();

    const d_mm = try da.bmm(db, .{});
    defer d_mm.deinit();
    _ = try d_mm.realize();

    // --- Lower & execute ---
    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();
    try ir.verify();

    var exec_cpu = zg.device.HostDevice.init();
    defer exec_cpu.deinit();
    const exec_dev = exec_cpu.reference();

    var input_map = std.AutoArrayHashMapUnmanaged(u32, []const f32).empty;
    defer input_map.deinit(std.testing.allocator);
    for (ir.input_ids.items) |input_id| {
        const value = ir.valueById(input_id).?;
        if (std.mem.eql(u8, value.label.?, "a")) {
            try input_map.put(std.testing.allocator, input_id, &a_data);
        } else {
            try input_map.put(std.testing.allocator, input_id, &b_data);
        }
    }

    var exec_result = try ir.execute(f32, std.testing.allocator, exec_dev, input_map);
    defer exec_result.deinit();

    for (ir.output_ids.items) |output_id| {
        const ir_output = exec_result.getOutput(output_id).?;
        try std.testing.expectEqualSlices(f32, eager_result, ir_output);
    }
}

test "graph_ir/execute roundtrip parity: multi-layer forward (matmul + add + relu)" {
    // Simulates a 2-layer MLP forward pass: relu(x @ W1 + b1) @ W2 + b2
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);
    const relu = zg.nn(f32).relu;

    // x=[1,3], W1=[3,4], b1=[1,4], W2=[4,2], b2=[1,2]
    const x_data = [_]f32{ 1.0, 0.5, -0.3 };
    const w1_data = [_]f32{ 0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3, 0.1, 0.2, -0.4 };
    const b1_data = [_]f32{ 0.01, -0.01, 0.02, 0.0 };
    const w2_data = [_]f32{ 0.5, -0.1, 0.2, 0.3, -0.4, 0.1, 0.6, -0.2 };
    const b2_data = [_]f32{ 0.1, -0.05 };

    // --- Eager reference ---
    var eager_cpu = zg.device.HostDevice.init();
    defer eager_cpu.deinit();
    const eager_dev = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const ex = try Tensor.from_slice(eager_dev, &x_data, &.{ 1, 3 }, .{ .graph = &eager_graph });
    defer ex.deinit();
    const ew1 = try Tensor.from_slice(eager_dev, &w1_data, &.{ 3, 4 }, .{ .graph = &eager_graph });
    defer ew1.deinit();
    const eb1 = try Tensor.from_slice(eager_dev, &b1_data, &.{ 1, 4 }, .{ .graph = &eager_graph });
    defer eb1.deinit();
    const ew2 = try Tensor.from_slice(eager_dev, &w2_data, &.{ 4, 2 }, .{ .graph = &eager_graph });
    defer ew2.deinit();
    const eb2 = try Tensor.from_slice(eager_dev, &b2_data, &.{ 1, 2 }, .{ .graph = &eager_graph });
    defer eb2.deinit();

    const h1 = try ex.bmm(ew1, .{});
    defer h1.deinit();
    const h1b = try h1.add(eb1);
    defer h1b.deinit();
    const h1r = try relu(h1b);
    defer h1r.deinit();
    const h2 = try h1r.bmm(ew2, .{});
    defer h2.deinit();
    const out = try h2.add(eb2);
    defer out.deinit();

    const eager_result = try out.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // --- Deferred capture ---
    var def_cpu = zg.device.HostDevice.init();
    defer def_cpu.deinit();
    const def_dev = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var cap = try session.begin();
    defer cap.end();

    const dx = try Tensor.from_slice(def_dev, &x_data, &.{ 1, 3 }, .{ .graph = &def_graph, .label = "x" });
    defer dx.deinit();
    const dw1 = try Tensor.from_slice(def_dev, &w1_data, &.{ 3, 4 }, .{ .graph = &def_graph, .label = "W1" });
    defer dw1.deinit();
    const db1 = try Tensor.from_slice(def_dev, &b1_data, &.{ 1, 4 }, .{ .graph = &def_graph, .label = "b1" });
    defer db1.deinit();
    const dw2 = try Tensor.from_slice(def_dev, &w2_data, &.{ 4, 2 }, .{ .graph = &def_graph, .label = "W2" });
    defer dw2.deinit();
    const db2 = try Tensor.from_slice(def_dev, &b2_data, &.{ 1, 2 }, .{ .graph = &def_graph, .label = "b2" });
    defer db2.deinit();

    const dh1 = try dx.bmm(dw1, .{});
    defer dh1.deinit();
    const dh1b = try dh1.add(db1);
    defer dh1b.deinit();
    const dh1r = try relu(dh1b);
    defer dh1r.deinit();
    const dh2 = try dh1r.bmm(dw2, .{});
    defer dh2.deinit();
    const dout = try dh2.add(db2);
    defer dout.deinit();
    _ = try dout.realize();

    // --- Lower to IR, execute ---
    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();
    try ir.verify();

    var exec_cpu = zg.device.HostDevice.init();
    defer exec_cpu.deinit();
    const exec_dev = exec_cpu.reference();

    // Build input map
    const label_data = [_]struct { label: []const u8, data: []const f32 }{
        .{ .label = "x", .data = &x_data },
        .{ .label = "W1", .data = &w1_data },
        .{ .label = "b1", .data = &b1_data },
        .{ .label = "W2", .data = &w2_data },
        .{ .label = "b2", .data = &b2_data },
    };

    var input_map = std.AutoArrayHashMapUnmanaged(u32, []const f32).empty;
    defer input_map.deinit(std.testing.allocator);
    for (ir.input_ids.items) |input_id| {
        const value = ir.valueById(input_id).?;
        for (label_data) |ld| {
            if (value.label != null and std.mem.eql(u8, value.label.?, ld.label)) {
                try input_map.put(std.testing.allocator, input_id, ld.data);
                break;
            }
        }
    }

    var exec_result = try ir.execute(f32, std.testing.allocator, exec_dev, input_map);
    defer exec_result.deinit();

    for (ir.output_ids.items) |output_id| {
        const ir_output = exec_result.getOutput(output_id).?;
        // Compare with tolerance for floating-point accumulation
        try std.testing.expectEqual(eager_result.len, ir_output.len);
        for (eager_result, ir_output) |expected, actual| {
            try std.testing.expectApproxEqAbs(expected, actual, 1e-5);
        }
    }
}

test "graph_ir/execute roundtrip parity: capture-optimize-execute" {
    // Full pipeline: capture → DCE → execute → verify parity
    const zg = @import("zigrad.zig");
    const Graph = zg.Graph;
    const Tensor = zg.NDTensor(f32);

    const a_data = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b_data = [_]f32{ 7, 8, 9, 10, 11, 12 };

    // --- Eager reference ---
    var eager_cpu = zg.device.HostDevice.init();
    defer eager_cpu.deinit();
    const eager_dev = eager_cpu.reference();

    var eager_graph = Graph.init(std.testing.allocator, .{});
    defer eager_graph.deinit();

    const ea = try Tensor.from_slice(eager_dev, &a_data, &.{ 2, 3 }, .{ .graph = &eager_graph });
    defer ea.deinit();
    const eb = try Tensor.from_slice(eager_dev, &b_data, &.{ 2, 3 }, .{ .graph = &eager_graph });
    defer eb.deinit();

    const e_sum = try ea.add(eb);
    defer e_sum.deinit();
    const e_prod = try e_sum.mul(ea);
    defer e_prod.deinit();

    const eager_result = try e_prod.to_host_owned(std.testing.allocator);
    defer std.testing.allocator.free(eager_result);

    // --- Deferred capture with dead branch ---
    var def_cpu = zg.device.HostDevice.init();
    defer def_cpu.deinit();
    const def_dev = def_cpu.reference();

    var def_graph = Graph.init(std.testing.allocator, .{});
    defer def_graph.deinit();

    var session = lazy.Session.init(std.testing.allocator);
    defer session.deinit();
    session.mode = .deferred;

    var cap = try session.begin();
    defer cap.end();

    const da = try Tensor.from_slice(def_dev, &a_data, &.{ 2, 3 }, .{ .graph = &def_graph, .label = "a" });
    defer da.deinit();
    const db = try Tensor.from_slice(def_dev, &b_data, &.{ 2, 3 }, .{ .graph = &def_graph, .label = "b" });
    defer db.deinit();

    const d_sum = try da.add(db);
    defer d_sum.deinit();
    const d_prod = try d_sum.mul(da);
    defer d_prod.deinit();
    // Dead branch
    const d_dead = try da.sub(db);
    defer d_dead.deinit();

    _ = try d_prod.realize();

    // --- Lower → optimize → execute ---
    var ir = try GraphIR.fromSession(std.testing.allocator, &session);
    defer ir.deinit();
    try ir.verify();

    // Should have 3 ops (add, mul, sub) before DCE
    try std.testing.expectEqual(@as(usize, 3), ir.opCount());

    // Run DCE
    var pm = PassManager.init(std.testing.allocator);
    defer pm.deinit();
    try pm.addPass(dcePass());
    try pm.run(&ir);

    // After DCE: dead sub removed
    try std.testing.expectEqual(@as(usize, 2), ir.opCount());

    var exec_cpu = zg.device.HostDevice.init();
    defer exec_cpu.deinit();
    const exec_dev = exec_cpu.reference();

    var input_map = std.AutoArrayHashMapUnmanaged(u32, []const f32).empty;
    defer input_map.deinit(std.testing.allocator);
    for (ir.input_ids.items) |input_id| {
        const value = ir.valueById(input_id).?;
        if (value.label != null and std.mem.eql(u8, value.label.?, "a")) {
            try input_map.put(std.testing.allocator, input_id, &a_data);
        } else {
            try input_map.put(std.testing.allocator, input_id, &b_data);
        }
    }

    var exec_result = try ir.execute(f32, std.testing.allocator, exec_dev, input_map);
    defer exec_result.deinit();

    for (ir.output_ids.items) |output_id| {
        const ir_output = exec_result.getOutput(output_id).?;
        try std.testing.expectEqualSlices(f32, eager_result, ir_output);
    }
}

test "graph_ir/topoSort returns valid execution order" {
    var ir = GraphIR.init(std.testing.allocator);
    defer ir.deinit();

    // Build a simple graph: input -> op1 -> op2 -> output
    // Values: 1 (input), 2 (op1 result), 3 (op2 result)
    const shape1 = try std.testing.allocator.dupe(usize, &.{4});
    try ir.values.append(std.testing.allocator, .{
        .id = 1, .dtype = .f32, .shape = shape1, .device = .host,
        .storage = .owned, .defining_op = null, .label = null, .requires_grad = false,
    });
    try ir.input_ids.append(std.testing.allocator, 1);

    const shape2 = try std.testing.allocator.dupe(usize, &.{4});
    try ir.values.append(std.testing.allocator, .{
        .id = 2, .dtype = .f32, .shape = shape2, .device = .host,
        .storage = .owned, .defining_op = 1, .label = null, .requires_grad = false,
    });

    const shape3 = try std.testing.allocator.dupe(usize, &.{4});
    try ir.values.append(std.testing.allocator, .{
        .id = 3, .dtype = .f32, .shape = shape3, .device = .host,
        .storage = .owned, .defining_op = 2, .label = null, .requires_grad = false,
    });
    try ir.output_ids.append(std.testing.allocator, 3);

    // Op1: input(1) -> result(2)
    const op1_operands = try std.testing.allocator.dupe(u32, &.{1});
    const op1_results = try std.testing.allocator.dupe(u32, &.{2});
    try ir.ops.append(std.testing.allocator, .{
        .id = 1, .name = "ADD", .operands = op1_operands,
        .results = op1_results, .attributes = &.{},
    });

    // Op2: result(2) -> result(3)
    const op2_operands = try std.testing.allocator.dupe(u32, &.{2});
    const op2_results = try std.testing.allocator.dupe(u32, &.{3});
    try ir.ops.append(std.testing.allocator, .{
        .id = 2, .name = "EXP", .operands = op2_operands,
        .results = op2_results, .attributes = &.{},
    });

    const sorted = try ir.topoSort(std.testing.allocator);
    defer std.testing.allocator.free(sorted);

    // Op1 must come before Op2
    try std.testing.expectEqual(@as(usize, 2), sorted.len);
    try std.testing.expectEqual(@as(u32, 1), sorted[0]);
    try std.testing.expectEqual(@as(u32, 2), sorted[1]);
}
