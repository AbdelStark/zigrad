const std = @import("std");
const zg = @import("zigrad");
const std_options = .{ .log_level = .info };
const log = std.log.scoped(.hello_world);
const T = f32;

pub const zigrad_settings: zg.Settings = .{
    .thread_safe = false,
    // zigrad is opt-in logging, logging scopes that
    // are not specified will not be executed.
    .logging = .{
        .level = .debug,
        .scopes = &.{
            .{ .scope = .zg_block_pool, .level = .debug },
            .{ .scope = .zg_caching_allocator, .level = .debug },
        },
    },
};

pub fn main() !void {
    const allocator = std.heap.smp_allocator;

    // Zigrad has a global graph that can be
    // overriden for user-provided graphs.
    zg.global_graph_init(allocator, .{
        .eager_teardown = true,
    });
    defer zg.global_graph_deinit();

    // Create a device that will provide blas,
    // tensor, and nn functionality.
    var runtime_device = try zg.device.initRuntimeDevice(null, .{ .allow_cuda = true });
    defer runtime_device.deinit();
    _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "hello-world:start",
        .include_telemetry = false,
    }) catch {};
    defer _ = runtime_device.maybeWriteRuntimeDiagnostics(std.fs.File.stderr().deprecatedWriter(), .{
        .label = "hello-world:summary",
        .include_telemetry = true,
    }) catch {};

    // Generic device interface (union of pointers)
    const device = runtime_device.reference();

    // Create a uniformly random tensor
    const x = try zg.NDTensor(f32).random(device, &.{ 32, 32 }, .uniform, .{});
    defer x.deinit();

    const y = try zg.NDTensor(f32).random(device, &.{ 32, 32 }, .uniform, .{});
    defer y.deinit();

    const z = try x.add(y);
    defer z.deinit();
}
