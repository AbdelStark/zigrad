const std = @import("std");

pub fn BoundedArray(comptime T: type, comptime capacity: usize) type {
    return struct {
        const Self = @This();

        buffer: [capacity]T = undefined,
        len: usize = 0,

        pub fn fromSlice(values: []const T) error{Overflow}!Self {
            if (values.len > capacity) return error.Overflow;

            var self: Self = .{ .len = values.len };
            @memcpy(self.buffer[0..values.len], values);
            return self;
        }

        pub fn slice(self: *const Self) []const T {
            return self.buffer[0..self.len];
        }

        pub fn get(self: *const Self, index: usize) T {
            return self.buffer[index];
        }

        pub fn append(self: *Self, value: T) error{Overflow}!void {
            if (self.len == capacity) return error.Overflow;

            self.buffer[self.len] = value;
            self.len += 1;
        }

        pub fn appendSlice(self: *Self, values: []const T) error{Overflow}!void {
            if (self.len + values.len > capacity) return error.Overflow;

            @memcpy(self.buffer[self.len .. self.len + values.len], values);
            self.len += values.len;
        }

        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;

            self.len -= 1;
            return self.buffer[self.len];
        }

        pub fn orderedRemove(self: *Self, index: usize) T {
            const value = self.buffer[index];
            @memmove(self.buffer[index .. self.len - 1], self.buffer[index + 1 .. self.len]);
            self.len -= 1;
            return value;
        }

        pub const Writer = if (T != u8)
            @compileError("writer() is only available for BoundedArray(u8, ...)")
        else
            std.io.GenericWriter(*Self, error{Overflow}, appendWrite);

        pub fn writer(self: *Self) Writer {
            return .{ .context = self };
        }

        fn appendWrite(self: *Self, bytes: []const u8) error{Overflow}!usize {
            try self.appendSlice(bytes);
            return bytes.len;
        }
    };
}
