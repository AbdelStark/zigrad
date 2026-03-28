const std = @import("std");

pub fn ReplayBuffer(T: type) type {
    return struct {
        const Self = @This();
        const Sample = std.MultiArrayList(T);

        allocator: std.mem.Allocator,
        data: []T,
        size: usize = 0,
        next_index: usize = 0,

        pub fn init(allocator: std.mem.Allocator, buffer_capacity: usize) !Self {
            if (buffer_capacity == 0) return error.InvalidReplayCapacity;
            return .{
                .allocator = allocator,
                .data = try allocator.alloc(T, buffer_capacity),
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
            self.* = undefined;
        }

        pub fn add(self: *Self, value: T) void {
            self.data[self.next_index] = value;
            self.size = @min(self.size + 1, self.data.len);
            self.next_index = (self.next_index + 1) % self.data.len;
        }

        pub fn capacity(self: *const Self) usize {
            return self.data.len;
        }

        pub fn sampleWithoutReplacement(
            self: *const Self,
            allocator: std.mem.Allocator,
            rng: std.Random,
            count: usize,
        ) !Sample {
            if (count == 0) return error.InvalidSampleSize;
            if (self.size < count) return error.InsufficientSamples;

            var indices = try allocator.alloc(usize, self.size);
            defer allocator.free(indices);
            for (0..self.size) |index| {
                indices[index] = index;
            }

            var sample = Sample{};
            errdefer sample.deinit(allocator);

            for (0..count) |index| {
                const swap_index = index + rng.uintLessThan(usize, self.size - index);
                std.mem.swap(usize, &indices[index], &indices[swap_index]);
                try sample.append(allocator, self.data[indices[index]]);
            }

            return sample;
        }
    };
}

test "replay buffer samples without duplicate indices" {
    const Entry = struct {
        index: usize,
    };

    var buffer = try ReplayBuffer(Entry).init(std.testing.allocator, 4);
    defer buffer.deinit();

    for (0..4) |index| {
        buffer.add(.{ .index = index });
    }

    var prng = std.Random.DefaultPrng.init(42);
    var sample = try buffer.sampleWithoutReplacement(std.testing.allocator, prng.random(), 4);
    defer sample.deinit(std.testing.allocator);

    var seen = [_]bool{ false, false, false, false };
    for (sample.items(.index)) |index| {
        try std.testing.expect(index < seen.len);
        try std.testing.expect(!seen[index]);
        seen[index] = true;
    }

    for (seen) |hit| {
        try std.testing.expect(hit);
    }
}
