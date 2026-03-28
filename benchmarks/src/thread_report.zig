const std = @import("std");
const result = @import("result.zig");

pub const InputFile = struct {
    path: []const u8,
    records: []const result.Record,
};

pub const ThreadEntry = struct {
    thread_count: ?u32,
    input_path: []const u8,
    status: result.Status,
    mean_ns: ?f64 = null,
    throughput_per_second: ?f64 = null,
    throughput_unit: ?[]const u8 = null,
    delta_ratio_vs_baseline: ?f64 = null,
    speedup_vs_baseline: ?f64 = null,
    efficiency_vs_baseline: ?f64 = null,
    peak_live_bytes: ?u64 = null,
    peak_graph_arena_bytes: ?u64 = null,
    peak_scratch_bytes: ?u64 = null,
    telemetry: ?result.HostBlasTelemetry = null,
    notes: ?[]const u8 = null,
};

pub const BenchmarkGroup = struct {
    benchmark_id: []const u8,
    suite: []const u8,
    kind: []const u8,
    dtype: []const u8,
    runner: []const u8,
    provider: []const u8,
    baseline_thread_count: ?u32 = null,
    baseline_available: bool = false,
    entries: []const ThreadEntry,
};

pub const Summary = struct {
    input_files: usize = 0,
    host_records: usize = 0,
    benchmark_groups: usize = 0,
    provider_count: usize = 0,
    ok_records: usize = 0,
    skipped_records: usize = 0,
    failed_records: usize = 0,
    baseline_missing_groups: usize = 0,
    comparable_entries: usize = 0,
};

pub const Report = struct {
    input_paths: []const []const u8,
    providers: []const []const u8,
    runner_filter: ?[]const u8,
    baseline_thread_count: ?u32 = null,
    summary: Summary,
    groups: []const BenchmarkGroup,
};

pub const Options = struct {
    input_paths: []const []const u8,
    runner_filter: ?[]const u8 = "zig",
    baseline_thread_count: ?u32 = null,
    markdown_output_path: ?[]const u8 = null,
    json_output_path: ?[]const u8 = null,
};

const GroupBuilder = struct {
    benchmark_id: []const u8,
    suite: []const u8,
    kind: []const u8,
    dtype: []const u8,
    runner: []const u8,
    provider: []const u8,
    shapes: []const result.ShapeMetadata,
    batch_size: ?usize,
    entries: std.ArrayList(ThreadEntry),

    fn init(record_entry: result.Record) GroupBuilder {
        return .{
            .benchmark_id = record_entry.benchmark_id,
            .suite = record_entry.suite,
            .kind = record_entry.kind,
            .dtype = record_entry.dtype,
            .runner = record_entry.runner,
            .provider = record_entry.backend.host_provider,
            .shapes = record_entry.shapes,
            .batch_size = record_entry.batch_size,
            .entries = .{},
        };
    }

    fn deinit(self: *GroupBuilder, allocator: std.mem.Allocator) void {
        self.entries.deinit(allocator);
        self.* = undefined;
    }
};

const BaselineSelection = struct {
    thread_count: ?u32 = null,
    available: bool = false,
};

const TestRecordSpec = struct {
    benchmark_id: []const u8,
    provider: []const u8,
    runner: []const u8 = "zig",
    status: result.Status = .ok,
    mean_ns: ?f64 = 100.0,
    throughput_per_second: ?f64 = 10.0,
    thread_count: ?u32 = 1,
    peak_live_bytes: ?u64 = null,
    peak_graph_arena_bytes: ?u64 = null,
    peak_scratch_bytes: ?u64 = null,
    telemetry: ?result.HostBlasTelemetry = null,
    notes: ?[]const u8 = null,
};

pub fn runCli(allocator: std.mem.Allocator) !void {
    const options = try parseArgs(allocator);
    if (options.input_paths.len == 0) return error.MissingInputPath;

    var loaded_files = try allocator.alloc(result.LoadedFile, options.input_paths.len);
    var loaded_count: usize = 0;
    defer {
        while (loaded_count > 0) : (loaded_count -= 1) {
            loaded_files[loaded_count - 1].deinit();
        }
    }

    var inputs = try allocator.alloc(InputFile, options.input_paths.len);
    for (options.input_paths, 0..) |path, idx| {
        loaded_files[idx] = try result.LoadedFile.loadFromFile(allocator, path);
        loaded_count += 1;
        inputs[idx] = .{
            .path = path,
            .records = loaded_files[idx].records,
        };
    }

    const report = try buildReport(
        allocator,
        inputs,
        options.runner_filter,
        options.baseline_thread_count,
    );

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try writeMarkdownReport(allocator, stdout, report);
    try stdout.flush();

    if (options.markdown_output_path) |path| {
        try writeMarkdownReportToPath(allocator, path, report);
    }
    if (options.json_output_path) |path| {
        try writeJsonReportToPath(path, report);
    }
}

pub fn buildReport(
    allocator: std.mem.Allocator,
    inputs: []const InputFile,
    runner_filter: ?[]const u8,
    requested_baseline_thread_count: ?u32,
) !Report {
    var group_builders = std.ArrayList(GroupBuilder){};
    errdefer {
        for (group_builders.items) |*group_builder| {
            group_builder.deinit(allocator);
        }
        group_builders.deinit(allocator);
    }

    var group_index = std.StringHashMap(usize).init(allocator);
    defer group_index.deinit();

    var provider_set = std.StringHashMap(void).init(allocator);
    defer provider_set.deinit();

    var input_paths = try allocator.alloc([]const u8, inputs.len);
    for (inputs, 0..) |input_file, idx| {
        input_paths[idx] = input_file.path;
    }

    var summary = Summary{
        .input_files = inputs.len,
    };

    for (inputs) |input_file| {
        for (input_file.records) |record_entry| {
            if (!runnerMatches(record_entry, runner_filter)) continue;
            if (!std.mem.eql(u8, record_entry.backend.device, "host")) continue;

            summary.host_records += 1;
            switch (record_entry.status) {
                .ok => summary.ok_records += 1,
                .skipped => summary.skipped_records += 1,
                .failed => summary.failed_records += 1,
            }

            try provider_set.put(record_entry.backend.host_provider, {});

            const key = try makeGroupKey(allocator, record_entry);
            const gop = try group_index.getOrPut(key);
            if (!gop.found_existing) {
                try group_builders.append(allocator, GroupBuilder.init(record_entry));
                gop.value_ptr.* = group_builders.items.len - 1;
            }

            var group_builder = &group_builders.items[gop.value_ptr.*];
            if (!recordMatchesGroup(group_builder.*, record_entry)) {
                return error.InconsistentBenchmarkMetadata;
            }

            for (group_builder.entries.items) |existing| {
                if (optionalThreadCountsEqual(existing.thread_count, record_entry.backend.thread_count)) {
                    return error.DuplicateThreadCountRecord;
                }
            }

            try group_builder.entries.append(allocator, .{
                .thread_count = record_entry.backend.thread_count,
                .input_path = input_file.path,
                .status = record_entry.status,
                .mean_ns = if (record_entry.stats) |stats| stats.mean_ns else null,
                .throughput_per_second = if (record_entry.stats) |stats| stats.throughput_per_second else null,
                .throughput_unit = if (record_entry.stats) |stats| stats.throughput_unit else null,
                .peak_live_bytes = if (record_entry.memory) |memory| memory.peak_live_bytes else null,
                .peak_graph_arena_bytes = if (record_entry.memory) |memory| memory.peak_graph_arena_bytes else null,
                .peak_scratch_bytes = if (record_entry.memory) |memory| memory.peak_scratch_bytes else null,
                .telemetry = record_entry.backend.host_blas_telemetry,
                .notes = record_entry.notes,
            });
        }
    }

    if (summary.host_records == 0) return error.NoHostThreadRecords;

    const providers = try collectProviders(allocator, &provider_set);
    summary.provider_count = providers.len;

    const groups = try allocator.alloc(BenchmarkGroup, group_builders.items.len);
    for (group_builders.items, 0..) |*group_builder, group_idx| {
        sortEntries(group_builder.entries.items);
        const baseline = applyBaseline(group_builder.entries.items, requested_baseline_thread_count);
        if (!baseline.available) summary.baseline_missing_groups += 1;
        for (group_builder.entries.items) |entry| {
            if (entry.delta_ratio_vs_baseline != null) summary.comparable_entries += 1;
        }

        groups[group_idx] = .{
            .benchmark_id = group_builder.benchmark_id,
            .suite = group_builder.suite,
            .kind = group_builder.kind,
            .dtype = group_builder.dtype,
            .runner = group_builder.runner,
            .provider = group_builder.provider,
            .baseline_thread_count = baseline.thread_count,
            .baseline_available = baseline.available,
            .entries = try group_builder.entries.toOwnedSlice(allocator),
        };
    }

    sortGroups(groups);
    group_builders.clearRetainingCapacity();
    group_builders.deinit(allocator);

    summary.benchmark_groups = groups.len;

    return .{
        .input_paths = input_paths,
        .providers = providers,
        .runner_filter = runner_filter,
        .baseline_thread_count = requested_baseline_thread_count,
        .summary = summary,
        .groups = groups,
    };
}

pub fn writeMarkdownReport(
    allocator: std.mem.Allocator,
    writer: anytype,
    report: Report,
) !void {
    try writer.writeAll("# Host Thread Scaling Report\n\n");
    try writer.writeAll("## Inputs\n");
    for (report.input_paths) |path| {
        try writer.print("- `{s}`\n", .{path});
    }
    try writer.writeByte('\n');

    try writer.writeAll("## Summary\n");
    try writer.print("- Host records: {}\n", .{report.summary.host_records});
    try writer.print("- Benchmark groups: {}\n", .{report.summary.benchmark_groups});
    try writer.print("- Providers: ", .{});
    for (report.providers, 0..) |provider, idx| {
        if (idx != 0) try writer.writeAll(", ");
        try writer.print("`{s}`", .{provider});
    }
    try writer.writeByte('\n');
    if (report.runner_filter) |runner| {
        try writer.print("- Runner filter: `{s}`\n", .{runner});
    }
    if (report.baseline_thread_count) |thread_count| {
        try writer.print("- Requested baseline thread count: `{d}`\n", .{thread_count});
    } else {
        try writer.writeAll("- Requested baseline thread count: smallest available per group\n");
    }
    try writer.print("- Groups missing a usable baseline: {}\n", .{report.summary.baseline_missing_groups});
    try writer.print("- Comparable entries with scaling deltas: {}\n", .{report.summary.comparable_entries});
    try writer.writeByte('\n');

    for (report.groups) |group| {
        try writer.print("## `{s}` on `{s}`\n\n", .{ group.benchmark_id, group.provider });
        try writer.print(
            "Suite `{s}`, kind `{s}`, dtype `{s}`, runner `{s}`. Baseline threads: {s}.\n\n",
            .{
                group.suite,
                group.kind,
                group.dtype,
                group.runner,
                try formatThreadCount(allocator, group.baseline_thread_count),
            },
        );

        try writer.writeAll("| Threads | Status | Mean ms | Delta vs baseline | Speedup vs baseline | Scaling efficiency | Throughput | Peak live KiB | Telemetry | Notes |\n");
        try writer.writeAll("| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |\n");
        for (group.entries) |entry| {
            try writer.print(
                "| {s} | `{s}` | {s} | {s} | {s} | {s} | {s} | {s} | {s} | {s} |\n",
                .{
                    try formatThreadCount(allocator, entry.thread_count),
                    @tagName(entry.status),
                    try formatMeanMs(allocator, entry.mean_ns),
                    try formatDeltaCell(allocator, group, entry),
                    try formatSpeedupCell(allocator, group, entry),
                    try formatEfficiencyCell(allocator, group, entry),
                    try formatThroughput(allocator, entry),
                    try formatKiB(allocator, entry.peak_live_bytes),
                    try formatTelemetry(allocator, entry.telemetry),
                    entry.notes orelse "-",
                },
            );
        }
        try writer.writeByte('\n');
    }
}

fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    var input_paths = std.ArrayList([]const u8){};
    errdefer input_paths.deinit(allocator);

    var options = Options{
        .input_paths = &.{},
    };

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];

        if (std.mem.eql(u8, arg, "--input")) {
            try input_paths.append(allocator, try nextArg(args, &index, "--input"));
        } else if (std.mem.eql(u8, arg, "--runner")) {
            options.runner_filter = try nextArg(args, &index, "--runner");
        } else if (std.mem.eql(u8, arg, "--baseline-thread-count")) {
            const value = try nextArg(args, &index, "--baseline-thread-count");
            options.baseline_thread_count = try std.fmt.parseInt(u32, value, 10);
            if (options.baseline_thread_count.? == 0) return error.InvalidThreadCount;
        } else if (std.mem.eql(u8, arg, "--markdown-output")) {
            options.markdown_output_path = try nextArg(args, &index, "--markdown-output");
        } else if (std.mem.eql(u8, arg, "--json-output")) {
            options.json_output_path = try nextArg(args, &index, "--json-output");
        } else if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return error.HelpPrinted;
        } else {
            return error.UnknownArgument;
        }
    }

    options.input_paths = try input_paths.toOwnedSlice(allocator);
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: benchmark-thread-report --input <path> [--input <path> ...] [--runner <name>] [--baseline-thread-count <n>] [--markdown-output <path>] [--json-output <path>]
        \\
    , .{});
}

fn nextArg(args: []const []const u8, index: *usize, flag: []const u8) ![]const u8 {
    index.* += 1;
    if (index.* >= args.len) {
        std.log.err("Missing value for {s}", .{flag});
        return error.MissingArgumentValue;
    }
    return args[index.*];
}

fn runnerMatches(record_entry: result.Record, runner_filter: ?[]const u8) bool {
    if (runner_filter) |runner| return std.mem.eql(u8, record_entry.runner, runner);
    return true;
}

fn makeGroupKey(allocator: std.mem.Allocator, record_entry: result.Record) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}\x1f{s}\x1f{s}",
        .{
            record_entry.benchmark_id,
            record_entry.runner,
            record_entry.backend.host_provider,
        },
    );
}

fn recordMatchesGroup(group_builder: GroupBuilder, record_entry: result.Record) bool {
    return std.mem.eql(u8, group_builder.suite, record_entry.suite) and
        std.mem.eql(u8, group_builder.kind, record_entry.kind) and
        std.mem.eql(u8, group_builder.dtype, record_entry.dtype) and
        optionalUsizeEqual(group_builder.batch_size, record_entry.batch_size) and
        shapesEqual(group_builder.shapes, record_entry.shapes);
}

fn optionalUsizeEqual(lhs: ?usize, rhs: ?usize) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return lhs.? == rhs.?;
}

fn optionalThreadCountsEqual(lhs: ?u32, rhs: ?u32) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return lhs.? == rhs.?;
}

fn shapesEqual(lhs: []const result.ShapeMetadata, rhs: []const result.ShapeMetadata) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_shape, rhs_shape| {
        if (!std.mem.eql(u8, lhs_shape.name, rhs_shape.name)) return false;
        if (!std.mem.eql(usize, lhs_shape.dims, rhs_shape.dims)) return false;
    }
    return true;
}

fn collectProviders(
    allocator: std.mem.Allocator,
    provider_set: *std.StringHashMap(void),
) ![]const []const u8 {
    var providers = std.ArrayList([]const u8){};
    errdefer providers.deinit(allocator);

    var iterator = provider_set.iterator();
    while (iterator.next()) |entry| {
        try providers.append(allocator, entry.key_ptr.*);
    }

    const slice = try providers.toOwnedSlice(allocator);
    std.mem.sort([]const u8, slice, {}, lessThanString);
    return slice;
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}

fn sortEntries(entries: []ThreadEntry) void {
    std.mem.sort(ThreadEntry, entries, {}, struct {
        fn lessThan(_: void, lhs: ThreadEntry, rhs: ThreadEntry) bool {
            return compareThreadCounts(lhs.thread_count, rhs.thread_count) == .lt;
        }
    }.lessThan);
}

fn applyBaseline(entries: []ThreadEntry, requested_baseline_thread_count: ?u32) BaselineSelection {
    var baseline_index: ?usize = null;

    if (requested_baseline_thread_count) |thread_count| {
        for (entries, 0..) |entry, idx| {
            if (entry.thread_count == null or entry.thread_count.? != thread_count) continue;
            if (entry.status != .ok or entry.mean_ns == null) {
                return .{
                    .thread_count = thread_count,
                    .available = false,
                };
            }
            baseline_index = idx;
            break;
        }
        if (baseline_index == null) {
            return .{
                .thread_count = thread_count,
                .available = false,
            };
        }
    } else {
        for (entries, 0..) |entry, idx| {
            if (entry.status != .ok or entry.mean_ns == null) continue;
            if (entry.thread_count == null) continue;
            baseline_index = idx;
            break;
        }
        if (baseline_index == null) {
            for (entries, 0..) |entry, idx| {
                if (entry.status == .ok and entry.mean_ns != null) {
                    baseline_index = idx;
                    break;
                }
            }
        }
    }

    if (baseline_index == null) return .{};

    const baseline_entry = entries[baseline_index.?];
    const baseline_mean_ns = baseline_entry.mean_ns.?;
    const baseline_thread_count = baseline_entry.thread_count;

    for (entries) |*entry| {
        if (entry.status != .ok or entry.mean_ns == null) continue;
        entry.delta_ratio_vs_baseline = percentageDelta(baseline_mean_ns, entry.mean_ns.?);
        entry.speedup_vs_baseline = baseline_mean_ns / entry.mean_ns.?;
        if (baseline_thread_count) |baseline_threads| {
            if (entry.thread_count) |entry_threads| {
                const ideal_scale = @as(f64, @floatFromInt(entry_threads)) /
                    @as(f64, @floatFromInt(baseline_threads));
                if (ideal_scale > 0.0) {
                    entry.efficiency_vs_baseline = entry.speedup_vs_baseline.? / ideal_scale;
                }
            }
        }
    }

    return .{
        .thread_count = baseline_thread_count,
        .available = true,
    };
}

fn percentageDelta(baseline_ns: f64, candidate_ns: f64) f64 {
    return (candidate_ns - baseline_ns) / baseline_ns;
}

fn sortGroups(groups: []BenchmarkGroup) void {
    std.mem.sort(BenchmarkGroup, groups, {}, struct {
        fn lessThan(_: void, lhs: BenchmarkGroup, rhs: BenchmarkGroup) bool {
            const benchmark_order = std.mem.order(u8, lhs.benchmark_id, rhs.benchmark_id);
            if (benchmark_order != .eq) return benchmark_order == .lt;

            const provider_order = std.mem.order(u8, lhs.provider, rhs.provider);
            if (provider_order != .eq) return provider_order == .lt;

            return std.mem.order(u8, lhs.runner, rhs.runner) == .lt;
        }
    }.lessThan);
}

fn compareThreadCounts(lhs: ?u32, rhs: ?u32) std.math.Order {
    if (lhs) |lhs_count| {
        if (rhs) |rhs_count| return std.math.order(lhs_count, rhs_count);
        return .gt;
    }
    if (rhs != null) return .lt;
    return .eq;
}

fn formatThreadCount(allocator: std.mem.Allocator, thread_count: ?u32) ![]const u8 {
    return if (thread_count) |count|
        std.fmt.allocPrint(allocator, "{d}", .{count})
    else
        allocator.dupe(u8, "n/a");
}

fn formatMeanMs(allocator: std.mem.Allocator, mean_ns: ?f64) ![]const u8 {
    return if (mean_ns) |value|
        try std.fmt.allocPrint(allocator, "{d:.3}", .{value / 1_000_000.0})
    else
        allocator.dupe(u8, "n/a");
}

fn formatDeltaCell(
    allocator: std.mem.Allocator,
    group: BenchmarkGroup,
    entry: ThreadEntry,
) ![]const u8 {
    if (group.baseline_available and optionalThreadCountsEqual(group.baseline_thread_count, entry.thread_count)) {
        return allocator.dupe(u8, "baseline");
    }
    return if (entry.delta_ratio_vs_baseline) |delta|
        std.fmt.allocPrint(allocator, "{s}{d:.2}%", .{ signPrefix(delta), @abs(delta) * 100.0 })
    else
        allocator.dupe(u8, "n/a");
}

fn formatSpeedupCell(
    allocator: std.mem.Allocator,
    group: BenchmarkGroup,
    entry: ThreadEntry,
) ![]const u8 {
    if (group.baseline_available and optionalThreadCountsEqual(group.baseline_thread_count, entry.thread_count)) {
        return allocator.dupe(u8, "1.00x");
    }
    return if (entry.speedup_vs_baseline) |speedup|
        std.fmt.allocPrint(allocator, "{d:.2}x", .{speedup})
    else
        allocator.dupe(u8, "n/a");
}

fn formatEfficiencyCell(
    allocator: std.mem.Allocator,
    group: BenchmarkGroup,
    entry: ThreadEntry,
) ![]const u8 {
    if (group.baseline_available and optionalThreadCountsEqual(group.baseline_thread_count, entry.thread_count)) {
        return allocator.dupe(u8, "1.00x");
    }
    return if (entry.efficiency_vs_baseline) |efficiency|
        std.fmt.allocPrint(allocator, "{d:.2}x", .{efficiency})
    else
        allocator.dupe(u8, "n/a");
}

fn signPrefix(value: f64) []const u8 {
    return if (value >= 0.0) "+" else "-";
}

fn formatThroughput(allocator: std.mem.Allocator, entry: ThreadEntry) ![]const u8 {
    if (entry.throughput_per_second) |throughput| {
        if (entry.throughput_unit) |unit| {
            return try std.fmt.allocPrint(allocator, "{d:.3} {s}/s", .{ throughput, unit });
        }
        return try std.fmt.allocPrint(allocator, "{d:.3}/s", .{throughput});
    }
    return allocator.dupe(u8, "n/a");
}

fn formatKiB(allocator: std.mem.Allocator, value: ?u64) ![]const u8 {
    return if (value) |bytes|
        std.fmt.allocPrint(allocator, "{d:.2}", .{@as(f64, @floatFromInt(bytes)) / 1024.0})
    else
        allocator.dupe(u8, "n/a");
}

fn formatTelemetry(
    allocator: std.mem.Allocator,
    telemetry: ?result.HostBlasTelemetry,
) ![]const u8 {
    if (telemetry) |value| {
        return try std.fmt.allocPrint(
            allocator,
            "dot={d}, matvec={d}, matmul={d}, direct_bmm={d}, fallback_bmm={d}/{d}",
            .{
                value.dot_calls,
                value.matvec_calls,
                value.matmul_calls,
                value.direct_bmm_dispatches,
                value.fallback_bmm_dispatches,
                value.fallback_bmm_batches,
            },
        );
    }
    return allocator.dupe(u8, "-");
}

fn writeMarkdownReportToPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    report: Report,
) !void {
    if (std.fs.path.dirname(path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try writeMarkdownReport(allocator, writer, report);
    try writer.flush();
}

fn writeJsonReportToPath(path: []const u8, report: Report) !void {
    if (std.fs.path.dirname(path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try std.json.Stringify.value(report, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

fn appendTestRecord(
    allocator: std.mem.Allocator,
    builder: *std.ArrayList(u8),
    spec: TestRecordSpec,
) !void {
    const stats_json = if (spec.mean_ns) |mean_ns|
        try std.fmt.allocPrint(
            allocator,
            "{{\"min_ns\":{d},\"median_ns\":{d},\"mean_ns\":{d:.1},\"p95_ns\":{d},\"max_ns\":{d},\"throughput_per_second\":{s},\"throughput_unit\":{s}}}",
            .{
                @as(u64, @intFromFloat(mean_ns)),
                @as(u64, @intFromFloat(mean_ns)),
                mean_ns,
                @as(u64, @intFromFloat(mean_ns)),
                @as(u64, @intFromFloat(mean_ns)),
                if (spec.throughput_per_second) |throughput| try std.fmt.allocPrint(allocator, "{d:.1}", .{throughput}) else "null",
                if (spec.throughput_per_second != null) "\"samples\"" else "null",
            },
        )
    else
        "null";

    const telemetry_json = if (spec.telemetry) |telemetry|
        try std.fmt.allocPrint(
            allocator,
            "{{\"dot_calls\":{d},\"matvec_calls\":{d},\"matmul_calls\":{d},\"bmm_acc_calls\":{d},\"direct_bmm_dispatches\":{d},\"fallback_bmm_dispatches\":{d},\"fallback_bmm_batches\":{d}}}",
            .{
                telemetry.dot_calls,
                telemetry.matvec_calls,
                telemetry.matmul_calls,
                telemetry.bmm_acc_calls,
                telemetry.direct_bmm_dispatches,
                telemetry.fallback_bmm_dispatches,
                telemetry.fallback_bmm_batches,
            },
        )
    else
        "null";

    const memory_json = if (spec.peak_live_bytes != null or
        spec.peak_graph_arena_bytes != null or
        spec.peak_scratch_bytes != null)
        try std.fmt.allocPrint(
            allocator,
            "{{\"peak_live_bytes\":{s},\"final_live_bytes\":null,\"peak_graph_arena_bytes\":{s},\"final_graph_arena_bytes\":null,\"peak_scratch_bytes\":{s}}}",
            .{
                if (spec.peak_live_bytes) |value| try std.fmt.allocPrint(allocator, "{d}", .{value}) else "null",
                if (spec.peak_graph_arena_bytes) |value| try std.fmt.allocPrint(allocator, "{d}", .{value}) else "null",
                if (spec.peak_scratch_bytes) |value| try std.fmt.allocPrint(allocator, "{d}", .{value}) else "null",
            },
        )
    else
        "null";

    try builder.writer(allocator).print(
        "{{\"benchmark_id\":\"{s}\",\"suite\":\"primitive\",\"kind\":\"primitive_add\",\"runner\":\"{s}\",\"status\":\"{s}\",\"dtype\":\"f32\",\"warmup_iterations\":1,\"measured_iterations\":2,\"batch_size\":null,\"seed\":1,\"shapes\":[{{\"name\":\"lhs\",\"dims\":[1]}}],\"runtime\":{{\"timestamp_unix_ms\":0,\"git_commit\":\"deadbeef\",\"git_dirty\":false,\"zig_version\":\"0.15.2\",\"harness_version\":\"0.1.0\"}},\"system\":{{\"os\":\"linux\",\"kernel\":\"test\",\"arch\":\"x86_64\",\"cpu_model\":\"cpu\",\"cpu_logical_cores\":1,\"total_memory_bytes\":null}},\"backend\":{{\"device\":\"host\",\"host_provider\":\"{s}\",\"thread_count\":{s},\"accelerator\":null,\"host_blas_telemetry\":{s}}},\"setup_latency_ns\":10,\"stats\":{s},\"memory\":{s},\"notes\":{s}}}\n",
        .{
            spec.benchmark_id,
            spec.runner,
            @tagName(spec.status),
            spec.provider,
            if (spec.thread_count) |thread_count| try std.fmt.allocPrint(allocator, "{d}", .{thread_count}) else "null",
            telemetry_json,
            stats_json,
            memory_json,
            if (spec.notes) |notes| try std.fmt.allocPrint(allocator, "\"{s}\"", .{notes}) else "null",
        },
    );
}

fn loadTestInput(
    parent_allocator: std.mem.Allocator,
    specs: []const TestRecordSpec,
) !result.LoadedFile {
    var builder = std.ArrayList(u8){};
    defer builder.deinit(parent_allocator);

    for (specs) |spec| {
        try appendTestRecord(parent_allocator, &builder, spec);
    }

    return result.LoadedFile.loadFromSlice(parent_allocator, builder.items);
}

test "thread report groups thread counts per provider and computes efficiency" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var accelerate = try loadTestInput(allocator, &.{
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "accelerate",
            .thread_count = 1,
            .mean_ns = 100_000_000.0,
            .throughput_per_second = 10.0,
            .telemetry = .{ .matmul_calls = 1 },
        },
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "accelerate",
            .thread_count = 4,
            .mean_ns = 40_000_000.0,
            .throughput_per_second = 25.0,
            .telemetry = .{ .matmul_calls = 1 },
        },
    });
    defer accelerate.deinit();

    var openblas = try loadTestInput(allocator, &.{
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "openblas",
            .thread_count = 1,
            .mean_ns = 120_000_000.0,
            .throughput_per_second = 8.0,
        },
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "openblas",
            .thread_count = 4,
            .mean_ns = 60_000_000.0,
            .throughput_per_second = 16.0,
        },
    });
    defer openblas.deinit();

    const report = try buildReport(
        allocator,
        &.{
            .{ .path = "accelerate.jsonl", .records = accelerate.records },
            .{ .path = "openblas.jsonl", .records = openblas.records },
        },
        "zig",
        null,
    );

    try std.testing.expectEqual(@as(usize, 2), report.groups.len);
    try std.testing.expectEqual(@as(usize, 2), report.summary.provider_count);
    try std.testing.expectEqual(@as(usize, 4), report.summary.comparable_entries);
    try std.testing.expectEqual(@as(u32, 1), report.groups[0].baseline_thread_count.?);
    try std.testing.expectEqual(@as(u32, 4), report.groups[0].entries[1].thread_count.?);
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), report.groups[0].entries[1].speedup_vs_baseline.?, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.625), report.groups[0].entries[1].efficiency_vs_baseline.?, 1e-9);
}

test "thread report rejects duplicate thread counts within a provider group" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var first = try loadTestInput(allocator, &.{
        .{ .benchmark_id = "primitive.matmul", .provider = "accelerate", .thread_count = 1 },
    });
    defer first.deinit();

    var second = try loadTestInput(allocator, &.{
        .{ .benchmark_id = "primitive.matmul", .provider = "accelerate", .thread_count = 1 },
    });
    defer second.deinit();

    try std.testing.expectError(
        error.DuplicateThreadCountRecord,
        buildReport(
            allocator,
            &.{
                .{ .path = "first.jsonl", .records = first.records },
                .{ .path = "second.jsonl", .records = second.records },
            },
            "zig",
            null,
        ),
    );
}

test "thread report markdown includes baseline speedup and efficiency" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var input = try loadTestInput(allocator, &.{
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "accelerate",
            .thread_count = 1,
            .mean_ns = 100_000_000.0,
            .throughput_per_second = 10.0,
            .telemetry = .{ .matmul_calls = 1 },
        },
        .{
            .benchmark_id = "primitive.matmul",
            .provider = "accelerate",
            .thread_count = 2,
            .mean_ns = 60_000_000.0,
            .throughput_per_second = 16.0,
            .telemetry = .{ .matmul_calls = 1, .direct_bmm_dispatches = 1 },
            .notes = "scaled run",
        },
    });
    defer input.deinit();

    const report = try buildReport(
        allocator,
        &.{
            .{ .path = "accelerate.jsonl", .records = input.records },
        },
        "zig",
        1,
    );

    var output = std.ArrayList(u8){};
    defer output.deinit(allocator);

    try writeMarkdownReport(allocator, output.writer(allocator), report);
    try std.testing.expect(std.mem.indexOf(u8, output.items, "| 1 | `ok` | 100.000 | baseline | 1.00x | 1.00x | 10.000 samples/s | n/a | dot=0, matvec=0, matmul=1, direct_bmm=0, fallback_bmm=0/0 | - |") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.items, "| 2 | `ok` | 60.000 | -40.00% | 1.67x | 0.83x | 16.000 samples/s | n/a | dot=0, matvec=0, matmul=1, direct_bmm=1, fallback_bmm=0/0 | scaled run |") != null);
}
