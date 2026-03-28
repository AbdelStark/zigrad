const std = @import("std");
const builtin = @import("builtin");
const zg = @import("zigrad");
const result = @import("result.zig");

pub const Snapshot = struct {
    runtime: result.RuntimeMetadata,
    system: result.SystemMetadata,
    backend: result.BackendMetadata,
};

pub fn collect(
    allocator: std.mem.Allocator,
    harness_version: []const u8,
    thread_count: ?u32,
    host_blas_telemetry: ?result.HostBlasTelemetry,
) !Snapshot {
    return .{
        .runtime = .{
            .timestamp_unix_ms = std.time.milliTimestamp(),
            .git_commit = try gitCommit(allocator),
            .git_dirty = try gitDirty(allocator),
            .zig_version = builtin.zig_version_string,
            .harness_version = harness_version,
        },
        .system = .{
            .os = @tagName(builtin.target.os.tag),
            .kernel = try kernelVersion(allocator),
            .arch = @tagName(builtin.target.cpu.arch),
            .cpu_model = try cpuModel(allocator),
            .cpu_logical_cores = std.Thread.getCpuCount() catch 0,
            .cpu_frequency_policy = cpuFrequencyPolicy(allocator),
            .total_memory_bytes = totalMemoryBytes(allocator),
        },
        .backend = .{
            .device = "host",
            .host_provider = hostProvider(),
            .thread_count = thread_count,
            .thread_environment = threadEnvironment(),
            .host_blas_telemetry = host_blas_telemetry,
        },
    };
}

pub fn hostProvider() []const u8 {
    _ = builtin;
    return zg.device.configured_host_blas_provider.name();
}

fn gitCommit(allocator: std.mem.Allocator) ![]const u8 {
    return runAndTrim(allocator, &.{ "git", "rev-parse", "HEAD" }, "unknown");
}

fn gitDirty(allocator: std.mem.Allocator) !bool {
    const status = try runAndTrim(allocator, &.{ "git", "status", "--porcelain" }, "");
    return status.len != 0;
}

fn kernelVersion(allocator: std.mem.Allocator) ![]const u8 {
    return runAndTrim(allocator, &.{ "uname", "-sr" }, "unknown");
}

fn cpuModel(allocator: std.mem.Allocator) ![]const u8 {
    return switch (builtin.target.os.tag) {
        .macos => blk: {
            const brand = try runAndTrim(allocator, &.{ "sysctl", "-n", "machdep.cpu.brand_string" }, "");
            if (brand.len != 0) break :blk brand;
            break :blk try runAndTrim(allocator, &.{ "sysctl", "-n", "hw.model" }, "unknown");
        },
        .linux => linuxCpuModel(allocator) catch try runAndTrim(allocator, &.{ "uname", "-m" }, "unknown"),
        else => "unknown",
    };
}

fn totalMemoryBytes(allocator: std.mem.Allocator) ?u64 {
    return switch (builtin.target.os.tag) {
        .macos => blk: {
            const raw = runAndTrim(allocator, &.{ "sysctl", "-n", "hw.memsize" }, "") catch break :blk null;
            break :blk parseU64(raw) catch null;
        },
        .linux => linuxMemTotal(allocator) catch null,
        else => null,
    };
}

fn cpuFrequencyPolicy(allocator: std.mem.Allocator) ?[]const u8 {
    return switch (builtin.target.os.tag) {
        .linux => linuxCpuFrequencyPolicy(allocator) catch null,
        else => null,
    };
}

fn threadEnvironment() ?result.ThreadEnvironment {
    const environment: result.ThreadEnvironment = .{
        .veclib_maximum_threads = getenv("VECLIB_MAXIMUM_THREADS"),
        .openblas_num_threads = getenv("OPENBLAS_NUM_THREADS"),
        .omp_num_threads = getenv("OMP_NUM_THREADS"),
        .mkl_num_threads = getenv("MKL_NUM_THREADS"),
        .mkl_dynamic = getenv("MKL_DYNAMIC"),
    };
    return if (environment.hasValues()) environment else null;
}

fn linuxCpuModel(allocator: std.mem.Allocator) ![]const u8 {
    const file = try std.fs.openFileAbsolute("/proc/cpuinfo", .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 20);

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "model name")) {
            const idx = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            return std.mem.trim(u8, line[idx + 1 ..], " \t");
        }
    }

    return error.CpuModelNotFound;
}

fn linuxMemTotal(allocator: std.mem.Allocator) !u64 {
    const file = try std.fs.openFileAbsolute("/proc/meminfo", .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 1 << 20);

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "MemTotal:")) {
            var it = std.mem.tokenizeScalar(u8, line, ' ');
            _ = it.next();
            const value = it.next() orelse return error.InvalidMeminfo;
            return try std.fmt.parseInt(u64, value, 10) * 1024;
        }
    }

    return error.InvalidMeminfo;
}

fn linuxCpuFrequencyPolicy(allocator: std.mem.Allocator) ![]const u8 {
    const file = try std.fs.openFileAbsolute("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", .{});
    defer file.close();
    const bytes = try file.readToEndAlloc(allocator, 256);

    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    if (trimmed.len == 0) return error.InvalidCpuFrequencyPolicy;
    return try allocator.dupe(u8, trimmed);
}

fn runAndTrim(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    fallback: []const u8,
) ![]const u8 {
    const child = std.process.Child.run(.{
        .allocator = allocator,
        .argv = argv,
    }) catch return fallback;
    defer allocator.free(child.stdout);
    defer allocator.free(child.stderr);

    const trimmed = std.mem.trim(u8, child.stdout, " \t\r\n");
    if (trimmed.len == 0) return fallback;
    return try allocator.dupe(u8, trimmed);
}

fn parseU64(value: anytype) !u64 {
    return std.fmt.parseInt(u64, value, 10);
}

fn getenv(name: []const u8) ?[]const u8 {
    return if (std.posix.getenv(name)) |value| value else null;
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "host provider reflects configured host backend" {
    try std.testing.expectEqualStrings(
        zg.device.configured_host_blas_provider.name(),
        hostProvider(),
    );
}

test "collect attaches host BLAS telemetry to backend metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const snapshot = try collect(arena.allocator(), "0.1.0", 4, .{
        .matmul_calls = 3,
        .bmm_acc_calls = 1,
        .direct_bmm_dispatches = 1,
    });

    try std.testing.expectEqual(@as(u64, 3), snapshot.backend.host_blas_telemetry.?.matmul_calls);
    try std.testing.expectEqual(@as(u64, 1), snapshot.backend.host_blas_telemetry.?.bmm_acc_calls);
    try std.testing.expectEqual(@as(u64, 1), snapshot.backend.host_blas_telemetry.?.direct_bmm_dispatches);
}

test "collect captures thread environment metadata" {
    const omp_name: [*:0]const u8 = "OMP_NUM_THREADS";
    const omp_value: [*:0]const u8 = "7";
    const mkl_name: [*:0]const u8 = "MKL_DYNAMIC";
    const mkl_value: [*:0]const u8 = "FALSE";
    _ = setenv(omp_name, omp_value, 1);
    defer _ = unsetenv(omp_name);
    _ = setenv(mkl_name, mkl_value, 1);
    defer _ = unsetenv(mkl_name);

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const snapshot = try collect(arena.allocator(), "0.1.0", 7, null);
    const thread_env = snapshot.backend.thread_environment.?;

    try std.testing.expectEqualStrings("7", thread_env.omp_num_threads.?);
    try std.testing.expectEqualStrings("FALSE", thread_env.mkl_dynamic.?);
}
