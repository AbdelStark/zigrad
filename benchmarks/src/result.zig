const std = @import("std");

pub const Status = enum {
    ok,
    skipped,
    failed,
};

pub const ShapeMetadata = struct {
    name: []const u8,
    dims: []const usize,
};

pub const RuntimeMetadata = struct {
    timestamp_unix_ms: i64,
    git_commit: []const u8,
    git_dirty: bool,
    zig_version: []const u8,
    harness_version: []const u8,
};

pub const SystemMetadata = struct {
    os: []const u8,
    kernel: []const u8,
    arch: []const u8,
    cpu_model: []const u8,
    cpu_logical_cores: usize,
    total_memory_bytes: ?u64 = null,
};

pub const BackendMetadata = struct {
    device: []const u8,
    host_provider: []const u8,
    thread_count: ?u32 = null,
    accelerator: ?[]const u8 = null,
};

pub const SummaryStats = struct {
    min_ns: u64,
    median_ns: u64,
    mean_ns: f64,
    p95_ns: u64,
    max_ns: u64,
    throughput_per_second: ?f64 = null,
    throughput_unit: ?[]const u8 = null,

    pub fn fromTimings(
        allocator: std.mem.Allocator,
        timings_ns: []const u64,
        throughput_items: ?usize,
        throughput_unit: ?[]const u8,
    ) !SummaryStats {
        if (timings_ns.len == 0) return error.EmptyTimingSeries;

        const sorted = try allocator.dupe(u64, timings_ns);
        defer allocator.free(sorted);
        insertionSort(sorted);

        var sum: f64 = 0;
        for (timings_ns) |timing_ns| {
            sum += @as(f64, @floatFromInt(timing_ns));
        }

        const median_ns = if (sorted.len % 2 == 1)
            sorted[sorted.len / 2]
        else
            (sorted[(sorted.len / 2) - 1] + sorted[sorted.len / 2]) / 2;

        const p95_index = if (sorted.len == 1) 0 else ((sorted.len - 1) * 95) / 100;
        const mean_ns = sum / @as(f64, @floatFromInt(timings_ns.len));

        var throughput_per_second: ?f64 = null;
        if (throughput_items) |items| {
            if (mean_ns > 0) {
                const seconds = mean_ns / @as(f64, std.time.ns_per_s);
                throughput_per_second = @as(f64, @floatFromInt(items)) / seconds;
            }
        }

        return .{
            .min_ns = sorted[0],
            .median_ns = median_ns,
            .mean_ns = mean_ns,
            .p95_ns = sorted[p95_index],
            .max_ns = sorted[sorted.len - 1],
            .throughput_per_second = throughput_per_second,
            .throughput_unit = if (throughput_per_second == null) null else throughput_unit,
        };
    }
};

pub const Record = struct {
    benchmark_id: []const u8,
    suite: []const u8,
    kind: []const u8,
    runner: []const u8,
    status: Status,
    dtype: []const u8,
    warmup_iterations: u32,
    measured_iterations: u32,
    batch_size: ?usize,
    seed: u64,
    shapes: []const ShapeMetadata,
    runtime: RuntimeMetadata,
    system: SystemMetadata,
    backend: BackendMetadata,
    setup_latency_ns: ?u64 = null,
    stats: ?SummaryStats = null,
    notes: ?[]const u8 = null,
};

pub fn writeJsonLine(writer: anytype, record: Record) !void {
    try std.json.Stringify.value(record, .{}, writer);
    try writer.writeByte('\n');
}

fn insertionSort(values: []u64) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const key = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > key) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = key;
    }
}

test "summary stats compute deterministic aggregates" {
    const allocator = std.testing.allocator;
    const stats = try SummaryStats.fromTimings(allocator, &.{ 5, 1, 3, 2, 4 }, 10, "samples");

    try std.testing.expectEqual(@as(u64, 1), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 3), stats.median_ns);
    try std.testing.expectEqual(@as(u64, 4), stats.p95_ns);
    try std.testing.expectEqual(@as(u64, 5), stats.max_ns);
    try std.testing.expectApproxEqAbs(@as(f64, 3), stats.mean_ns, 1e-9);
    try std.testing.expect(stats.throughput_per_second != null);
}
