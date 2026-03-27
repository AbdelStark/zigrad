const std = @import("std");
const benchmarking = @import("benchmarking");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    benchmarking.compare.runCli(arena.allocator()) catch |err| switch (err) {
        error.HelpPrinted => {},
        else => return err,
    };
}
