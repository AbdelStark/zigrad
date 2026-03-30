const std = @import("std");

pub const ExecutionMode = enum {
    /// Record-only: operations execute eagerly, capture observes.
    observe,
    /// Deferred: operations are queued and replayed on realize().
    deferred,
};

pub const ThunkBase = struct {
    execute_fn: *const fn (*ThunkBase) void,
    cleanup_fn: *const fn (*ThunkBase, std.mem.Allocator) void,

    pub fn execute(self: *ThunkBase) void {
        self.execute_fn(self);
    }

    pub fn cleanup(self: *ThunkBase, allocator: std.mem.Allocator) void {
        self.cleanup_fn(self, allocator);
    }
};

pub const DeviceKind = enum {
    host,
    cuda,
};

pub const StorageKind = enum {
    owned,
    view,
};

pub const MaterializationReason = enum {
    explicit,
    host_read,
};

pub const AttributeValue = union(enum) {
    bool: bool,
    int: i64,
    uint: u64,
    float: f64,
    string: []const u8,
    usize_list: []const usize,
    int_list: []const i64,
};

pub const OpAttribute = struct {
    key: []const u8,
    value: AttributeValue,
};

pub const TensorRecord = struct {
    id: u32,
    op_name: []const u8,
    dtype_name: []const u8,
    shape: []const usize,
    device: DeviceKind,
    requires_grad: bool,
    attached: bool,
    acquired: bool,
    storage: StorageKind,
    attributes: []const OpAttribute = &.{},
    label: ?[]const u8,
    parent_ids: []const u32,
};

pub const MaterializationRecord = struct {
    tensor_id: u32,
    reason: MaterializationReason,
};

pub const TensorCapture = struct {
    tensor_key: usize,
    parent_keys: []const usize = &.{},
    op_name: []const u8,
    dtype_name: []const u8,
    shape: []const usize,
    device: DeviceKind,
    requires_grad: bool,
    attached: bool,
    acquired: bool,
    storage: StorageKind = .owned,
    attributes: []const OpAttribute = &.{},
    label: ?[]const u8 = null,
};

threadlocal var active_session: ?*Session = null;

pub fn isCapturing() bool {
    return active_session != null;
}

pub fn isDeferred() bool {
    const session = active_session orelse return false;
    return session.mode == .deferred;
}

pub fn deferredAllocator() std.mem.Allocator {
    return active_session.?.allocator;
}

pub fn enqueueDeferredThunk(thunk: *ThunkBase) void {
    const session = active_session.?;
    session.thunks.append(session.allocator, thunk) catch
        @panic("OOM: cannot enqueue deferred thunk in lazy session");
}

/// Enqueue a deferred backward pass. The backward closure runs with
/// the session suspended so all gradient dispatch calls execute
/// immediately.
pub fn enqueueDeferredBackward(node_ptr: anytype) !void {
    const session = active_session orelse return;
    const NodeType = @TypeOf(node_ptr);

    const Thunk = struct {
        base: ThunkBase,
        node: NodeType,

        fn execute(base_ptr: *ThunkBase) void {
            const self: *@This() = @fieldParentPtr("base", base_ptr);
            // Suspend the deferred session so gradient dispatch calls
            // execute immediately within the backward traversal.
            const saved = suspendSession();
            defer restoreSession(saved);

            const graph = self.node.gb.promote();
            graph.backward(self.node) catch
                @panic("deferred backward failed");
        }

        fn cleanup(base_ptr: *ThunkBase, allocator: std.mem.Allocator) void {
            const self: *@This() = @fieldParentPtr("base", base_ptr);
            allocator.destroy(self);
        }
    };

    const thunk = try session.allocator.create(Thunk);
    thunk.* = .{
        .base = .{
            .execute_fn = Thunk.execute,
            .cleanup_fn = Thunk.cleanup,
        },
        .node = node_ptr,
    };
    enqueueDeferredThunk(&thunk.base);
}

/// Temporarily suspend the active session so dispatch calls execute
/// immediately (used by the execution bridge). Returns the suspended
/// session so it can be restored later.
pub fn suspendSession() ?*Session {
    const s = active_session;
    active_session = null;
    return s;
}

/// Restore a previously suspended session.
pub fn restoreSession(session: ?*Session) void {
    active_session = session;
}

pub fn flushIfDeferred() void {
    const session = active_session orelse return;
    if (session.mode == .deferred) {
        session.flush();
    }
}

pub fn maybeRecordTensor(capture: TensorCapture) !void {
    if (active_session) |session| {
        try session.recordTensor(capture);
    }
}

pub fn maybeRecordMaterialization(tensor_key: usize, reason: MaterializationReason) !void {
    if (active_session) |session| {
        try session.recordMaterialization(tensor_key, reason);
    }
}

pub const SessionDump = struct {
    tensors: []const TensorRecord,
    materializations: []const MaterializationRecord,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    tensor_ids: std.AutoArrayHashMapUnmanaged(usize, u32) = .empty,
    records: std.ArrayListUnmanaged(TensorRecord) = .empty,
    materializations: std.ArrayListUnmanaged(MaterializationRecord) = .empty,
    thunks: std.ArrayListUnmanaged(*ThunkBase) = .empty,
    active_depth: usize = 0,
    mode: ExecutionMode = .observe,

    pub fn init(allocator: std.mem.Allocator) Session {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Session) void {
        self.reset();
        self.tensor_ids.deinit(self.allocator);
        self.records.deinit(self.allocator);
        self.materializations.deinit(self.allocator);
        self.thunks.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn reset(self: *Session) void {
        self.clearThunks();
        for (self.records.items) |record| {
            self.allocator.free(record.shape);
            freeAttributes(self.allocator, record.attributes);
            self.allocator.free(record.parent_ids);
            if (record.label) |label| self.allocator.free(label);
        }
        self.records.clearRetainingCapacity();
        self.materializations.clearRetainingCapacity();
        self.tensor_ids.clearRetainingCapacity();
    }

    pub fn flush(self: *Session) void {
        for (self.thunks.items) |thunk| {
            thunk.execute();
        }
        self.clearThunks();
    }

    pub fn pendingThunkCount(self: *const Session) usize {
        return self.thunks.items.len;
    }

    fn clearThunks(self: *Session) void {
        for (self.thunks.items) |thunk| {
            thunk.cleanup(self.allocator);
        }
        self.thunks.clearRetainingCapacity();
    }

    pub fn begin(self: *Session) !Guard {
        if (active_session) |current| {
            if (current != self) return error.LazySessionAlreadyActive;
        } else {
            active_session = self;
        }

        self.active_depth += 1;
        return .{ .session = self };
    }

    fn endCapture(self: *Session) void {
        std.debug.assert(self.active_depth > 0);
        std.debug.assert(active_session == self);

        self.active_depth -= 1;
        if (self.active_depth == 0) {
            active_session = null;
        }
    }

    pub fn tensors(self: *const Session) []const TensorRecord {
        return self.records.items;
    }

    pub fn materializationEvents(self: *const Session) []const MaterializationRecord {
        return self.materializations.items;
    }

    pub fn tensorById(self: *const Session, id: u32) ?*const TensorRecord {
        if (id == 0 or id > self.records.items.len) return null;
        return &self.records.items[id - 1];
    }

    pub fn lookupTensorId(self: *const Session, tensor_key: usize) ?u32 {
        return self.tensor_ids.get(tensor_key);
    }

    pub fn dump(self: *const Session) SessionDump {
        return .{
            .tensors = self.records.items,
            .materializations = self.materializations.items,
        };
    }

    pub fn writeJson(self: *const Session, writer: *std.Io.Writer) !void {
        try std.json.Stringify.value(self.dump(), .{}, writer);
    }

    pub fn writeSummary(self: *const Session, writer: anytype) !void {
        try writer.print(
            "lazy session tensors={d} materializations={d}\n",
            .{ self.records.items.len, self.materializations.items.len },
        );
        for (self.records.items) |record| {
            try writer.print(
                "#{d} op={s} dtype={s} device={s} storage={s} grad={} attached={} acquired={}",
                .{
                    record.id,
                    record.op_name,
                    record.dtype_name,
                    @tagName(record.device),
                    @tagName(record.storage),
                    record.requires_grad,
                    record.attached,
                    record.acquired,
                },
            );
            try writer.writeAll(" shape=");
            try writeShape(writer, record.shape);
            if (record.attributes.len != 0) {
                try writer.writeAll(" attrs={");
                try writeAttributes(writer, record.attributes);
                try writer.writeByte('}');
            }
            if (record.parent_ids.len != 0) {
                try writer.writeAll(" parents=[");
                for (record.parent_ids, 0..) |parent_id, index| {
                    if (index != 0) try writer.writeAll(",");
                    try writer.print("{d}", .{parent_id});
                }
                try writer.writeAll("]");
            }
            if (record.label) |label| {
                try writer.print(" label={s}", .{label});
            }
            try writer.writeByte('\n');
        }
        for (self.materializations.items) |event| {
            try writer.print(
                "materialize tensor=#{d} reason={s}\n",
                .{ event.tensor_id, @tagName(event.reason) },
            );
        }
    }

    pub fn writeD2(self: *const Session, writer: anytype) !void {
        for (self.records.items) |record| {
            try writer.print("t{d}: \"{s} #{d}\\n{s} ", .{
                record.id,
                record.op_name,
                record.id,
                record.dtype_name,
            });
            try writeShape(writer, record.shape);
            try writer.print("\\n{s} {s}", .{
                @tagName(record.device),
                @tagName(record.storage),
            });
            if (record.requires_grad) try writer.writeAll(" grad");
            if (record.attributes.len != 0) {
                try writer.writeAll("\\nattrs=");
                try writeAttributes(writer, record.attributes);
            }
            if (record.label) |label| try writer.print("\\nlabel={s}", .{label});
            try writer.writeAll("\"\n");
        }

        for (self.records.items) |record| {
            for (record.parent_ids) |parent_id| {
                try writer.print("t{d} -> t{d}\n", .{ parent_id, record.id });
            }
        }

        for (self.materializations.items, 0..) |event, index| {
            try writer.print("m{d}: \"materialize {s}\"\n", .{
                index + 1,
                @tagName(event.reason),
            });
            try writer.print("t{d} -> m{d}\n", .{
                event.tensor_id,
                index + 1,
            });
        }
    }

    fn recordTensor(self: *Session, capture: TensorCapture) !void {
        if (self.tensor_ids.contains(capture.tensor_key)) return;

        const parent_ids = try self.allocator.alloc(u32, capture.parent_keys.len);
        errdefer self.allocator.free(parent_ids);
        for (capture.parent_keys, 0..) |parent_key, index| {
            parent_ids[index] = self.tensor_ids.get(parent_key) orelse return error.UnknownCapturedParent;
        }

        const shape_copy = try self.allocator.dupe(usize, capture.shape);
        errdefer self.allocator.free(shape_copy);

        const attributes_copy = try dupeAttributes(self.allocator, capture.attributes);
        errdefer freeAttributes(self.allocator, attributes_copy);

        const label_copy = if (capture.label) |label|
            try self.allocator.dupe(u8, label)
        else
            null;
        errdefer if (label_copy) |label| self.allocator.free(label);

        const next_id: u32 = @intCast(self.records.items.len + 1);
        try self.records.append(self.allocator, .{
            .id = next_id,
            .op_name = capture.op_name,
            .dtype_name = capture.dtype_name,
            .shape = shape_copy,
            .device = capture.device,
            .requires_grad = capture.requires_grad,
            .attached = capture.attached,
            .acquired = capture.acquired,
            .storage = capture.storage,
            .attributes = attributes_copy,
            .label = label_copy,
            .parent_ids = parent_ids,
        });
        try self.tensor_ids.put(self.allocator, capture.tensor_key, next_id);
    }

    fn recordMaterialization(self: *Session, tensor_key: usize, reason: MaterializationReason) !void {
        const tensor_id = self.tensor_ids.get(tensor_key) orelse return;
        try self.materializations.append(self.allocator, .{
            .tensor_id = tensor_id,
            .reason = reason,
        });
    }
};

pub const Guard = struct {
    session: *Session,
    active: bool = true,

    pub fn end(self: *Guard) void {
        if (!self.active) return;
        self.active = false;
        self.session.endCapture();
    }
};

// Public wrappers for cross-module use (graph_ir.zig)
pub fn dupeAttributesPublic(allocator: std.mem.Allocator, attributes: []const OpAttribute) ![]const OpAttribute {
    return dupeAttributes(allocator, attributes);
}

pub fn freeAttributesPublic(allocator: std.mem.Allocator, attributes: []const OpAttribute) void {
    freeAttributes(allocator, attributes);
}

pub fn writeAttributesPublic(writer: anytype, attributes: []const OpAttribute) !void {
    return writeAttributes(writer, attributes);
}

fn writeShape(writer: anytype, shape: []const usize) !void {
    try writeUsizeList(writer, shape, "x");
}

fn writeUsizeList(writer: anytype, values: []const usize, separator: []const u8) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(separator);
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

fn writeI64List(writer: anytype, values: []const i64) !void {
    try writer.writeByte('[');
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

fn writeAttributes(writer: anytype, attributes: []const OpAttribute) !void {
    for (attributes, 0..) |attribute, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("{s}=", .{attribute.key});
        switch (attribute.value) {
            .bool => |value| try writer.writeAll(if (value) "true" else "false"),
            .int => |value| try writer.print("{d}", .{value}),
            .uint => |value| try writer.print("{d}", .{value}),
            .float => |value| try writer.print("{d}", .{value}),
            .string => |value| try writer.writeAll(value),
            .usize_list => |value| try writeUsizeList(writer, value, ","),
            .int_list => |value| try writeI64List(writer, value),
        }
    }
}

fn dupeAttributes(allocator: std.mem.Allocator, attributes: []const OpAttribute) ![]const OpAttribute {
    const copied = try allocator.alloc(OpAttribute, attributes.len);
    errdefer allocator.free(copied);

    var copied_len: usize = 0;
    errdefer {
        for (copied[0..copied_len]) |attribute| {
            freeAttribute(allocator, attribute);
        }
    }

    for (attributes, 0..) |attribute, index| {
        copied[index] = try dupeAttribute(allocator, attribute);
        copied_len += 1;
    }
    return copied;
}

fn dupeAttribute(allocator: std.mem.Allocator, attribute: OpAttribute) !OpAttribute {
    return .{
        .key = try allocator.dupe(u8, attribute.key),
        .value = try dupeAttributeValue(allocator, attribute.value),
    };
}

fn dupeAttributeValue(allocator: std.mem.Allocator, value: AttributeValue) !AttributeValue {
    return switch (value) {
        .bool => |item| .{ .bool = item },
        .int => |item| .{ .int = item },
        .uint => |item| .{ .uint = item },
        .float => |item| .{ .float = item },
        .string => |item| .{ .string = try allocator.dupe(u8, item) },
        .usize_list => |item| .{ .usize_list = try allocator.dupe(usize, item) },
        .int_list => |item| .{ .int_list = try allocator.dupe(i64, item) },
    };
}

fn freeAttributes(allocator: std.mem.Allocator, attributes: []const OpAttribute) void {
    for (attributes) |attribute| {
        freeAttribute(allocator, attribute);
    }
    allocator.free(attributes);
}

fn freeAttribute(allocator: std.mem.Allocator, attribute: OpAttribute) void {
    allocator.free(attribute.key);
    switch (attribute.value) {
        .string => |value| allocator.free(value),
        .usize_list => |value| allocator.free(value),
        .int_list => |value| allocator.free(value),
        else => {},
    }
}
