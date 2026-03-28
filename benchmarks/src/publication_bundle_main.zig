const std = @import("std");
const publication_bundle = @import("benchmarking").publication_bundle;

pub fn main() !void {
    publication_bundle.runCli(std.heap.smp_allocator) catch |err| switch (err) {
        error.HelpPrinted => {},
        else => return err,
    };
}
