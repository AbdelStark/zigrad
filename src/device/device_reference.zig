const std = @import("std");
const HostDevice = @import("host_device.zig");
const ReduceType = @import("device_common.zig").ReduceType;
const SmaxType = @import("device_common.zig").SmaxType;
const RandType = @import("device_common.zig").RandType;
const TransferDirection = @import("device_common.zig").TransferDirection;
const EnabledDevicePointers = @import("enabled_devices.zig").EnabledDevicePointers;
const lazy = @import("../lazy.zig");
const Self = @This();

const DeviceData = @import("../allocators.zig").DeviceData;
const Error = @import("../allocators.zig").Error;

ptrs: EnabledDevicePointers,

pub fn dispatch(self: Self, params: anytype) void {
    if (lazy.isDeferred()) {
        const P = @TypeOf(params);
        const Thunk = DeferredDispatchThunk(P);
        const alloc = lazy.deferredAllocator();
        const thunk = alloc.create(Thunk) catch
            @panic("OOM: cannot create deferred dispatch thunk");
        thunk.* = .{
            .base = .{ .execute_fn = Thunk.doExecute, .cleanup_fn = Thunk.doCleanup },
            .params = if (comptime Thunk.hasMetadataSlices())
                Thunk.dupeMetadataSlices(params, alloc)
            else
                params,
            .device_ptrs = self.ptrs,
        };
        lazy.enqueueDeferredThunk(&thunk.base);
        return;
    }
    dispatchImmediate(self.ptrs, params);
}

fn dispatchImmediate(ptrs: EnabledDevicePointers, params: anytype) void {
    switch (ptrs) {
        inline else => |d| {
            const D = std.meta.Child(@TypeOf(d));
            const P = @TypeOf(params);
            if (comptime !@hasDecl(D, P.__name__)) {
                @panic("Unimplemented: " ++ @typeName(D) ++ ", " ++ P.__name__);
            } else {
                @field(D, P.__name__)(d, P.__type__, params);
            }
        },
    }
}

fn DeferredDispatchThunk(comptime P: type) type {
    return struct {
        const ThunkSelf = @This();
        base: lazy.ThunkBase,
        params: P,
        device_ptrs: EnabledDevicePointers,

        /// Deep-copy metadata slices (e.g. shape arrays) that may reference
        /// stack-local memory in the caller. Data slices (typed as the
        /// element type T) point to device-managed buffers and stay valid.
        fn dupeMetadataSlices(params: P, allocator: std.mem.Allocator) P {
            var result = params;
            inline for (std.meta.fields(P)) |field| {
                if (field.type == []const usize) {
                    @field(result, field.name) = allocator.dupe(
                        usize,
                        @field(params, field.name),
                    ) catch @panic("OOM: cannot duplicate deferred metadata slice");
                }
            }
            return result;
        }

        fn freeMetadataSlices(params: P, allocator: std.mem.Allocator) void {
            inline for (std.meta.fields(P)) |field| {
                if (field.type == []const usize) {
                    allocator.free(@field(params, field.name));
                }
            }
        }

        fn hasMetadataSlices() bool {
            inline for (std.meta.fields(P)) |field| {
                if (field.type == []const usize) return true;
            }
            return false;
        }

        fn doExecute(base_ptr: *lazy.ThunkBase) void {
            const self: *ThunkSelf = @fieldParentPtr("base", base_ptr);
            dispatchImmediate(self.device_ptrs, self.params);
        }

        fn doCleanup(base_ptr: *lazy.ThunkBase, allocator: std.mem.Allocator) void {
            const self: *ThunkSelf = @fieldParentPtr("base", base_ptr);
            if (comptime hasMetadataSlices()) {
                freeMetadataSlices(self.params, allocator);
            }
            allocator.destroy(self);
        }
    };
}

pub fn mem_cache_alloc(self: Self, T: type, n: usize) !DeviceData(T) {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_cache_alloc(T, n),
    };
}

pub fn mem_cache_free(self: Self, data: anytype) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_cache_free(data),
    };
}

pub fn mem_cache_dupe(self: Self, T: type, src: []const T) !DeviceData(T) {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_cache_dupe(T, src),
    };
}

pub fn mem_alloc(self: Self, T: type, n: usize) ![]T {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_alloc(T, n),
    };
}

pub fn mem_alloc_byte_mask(self: Self, n: usize) ![]u8 {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_alloc_byte_mask(n),
    };
}

pub fn mem_free(self: Self, slice: anytype) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_free(slice),
    };
}

pub fn mem_dupe(self: Self, T: type, slice: anytype) ![]T {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_dupe(T, slice),
    };
}

pub fn mem_scratch(self: Self, T: type, n: usize) []T {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_scratch(T, n),
    };
}

pub fn mem_fill(self: Self, T: type, slice: []T, value: T) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_fill(T, slice, value),
    };
}

pub fn mem_random(self: Self, T: type, slice: []T, op: RandType, rand: std.Random) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_random(T, slice, op, rand),
    };
}

pub fn mem_copy(self: Self, T: type, src: []const T, dst: []T) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_copy(T, src, dst),
    };
}

pub fn mem_sequence(self: Self, T: type, dst: []T, initial: T, step: T) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_sequence(T, dst, initial, step),
    };
}

pub fn mem_transfer(self: Self, T: type, src: []const T, dst: []T, direction: TransferDirection) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_transfer(T, src, dst, direction),
    };
}

pub fn mem_take(self: Self, T: type, src: []const T, idxs: []const usize, dst: []T) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.mem_take(T, src, idxs, dst),
    };
}

pub fn clear_cache(self: Self) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.clear_cache(),
    };
}

pub fn recordDirectBmmDispatch(self: Self) void {
    switch (self.ptrs) {
        inline else => |dev| {
            const D = std.meta.Child(@TypeOf(dev));
            if (comptime @hasDecl(D, "recordDirectBmmDispatch")) {
                dev.recordDirectBmmDispatch();
            }
        },
    }
}

pub fn recordFallbackBmmDispatch(self: Self, batch_count: usize) void {
    switch (self.ptrs) {
        inline else => |dev| {
            const D = std.meta.Child(@TypeOf(dev));
            if (comptime @hasDecl(D, "recordFallbackBmmDispatch")) {
                dev.recordFallbackBmmDispatch(batch_count);
            }
        },
    }
}

pub fn sync(self: Self) void {
    return switch (self.ptrs) {
        inline else => |dev| dev.sync(),
    };
}

pub fn is_compatible(self: Self, other: Self) bool {
    return std.meta.activeTag(self.ptrs) == std.meta.activeTag(other.ptrs);
}

pub fn is_host(self: Self) bool {
    return self.ptrs == .host;
}
