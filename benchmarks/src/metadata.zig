const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
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
            .total_memory_bytes = totalMemoryBytes(allocator),
        },
        .backend = .{
            .device = "host",
            .host_provider = hostProvider(),
            .thread_count = thread_count,
        },
    };
}

pub fn hostProvider() []const u8 {
    return switch (builtin.target.os.tag) {
        .macos => "accelerate",
        .linux => if (build_options.enable_mkl) "mkl" else "blas",
        else => "unknown",
    };
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
