const std = @import("std");
const result = @import("result.zig");

pub const Thresholds = struct {
    warn_ratio: f64 = 0.05,
    fail_ratio: f64 = 0.10,
};

pub const Classification = enum {
    improved,
    pass,
    warn,
    fail,
    skipped,
    missing_candidate,
    new_candidate,
};

pub const Comparison = struct {
    benchmark_id: []const u8,
    runner: []const u8,
    classification: Classification,
    baseline_status: ?result.Status = null,
    candidate_status: ?result.Status = null,
    baseline_mean_ns: ?f64 = null,
    candidate_mean_ns: ?f64 = null,
    latency_delta_ratio: ?f64 = null,
    baseline_throughput_per_second: ?f64 = null,
    candidate_throughput_per_second: ?f64 = null,
    throughput_delta_ratio: ?f64 = null,
    notes: ?[]const u8 = null,
};

pub const Summary = struct {
    baseline_records: usize = 0,
    candidate_records: usize = 0,
    paired_records: usize = 0,
    improved: usize = 0,
    passing: usize = 0,
    warned: usize = 0,
    failed: usize = 0,
    skipped: usize = 0,
    missing_candidate: usize = 0,
    new_candidate: usize = 0,
    should_fail: bool = false,
};

pub const Report = struct {
    baseline_path: []const u8,
    candidate_path: []const u8,
    runner_filter: ?[]const u8,
    thresholds: Thresholds,
    summary: Summary,
    comparisons: []const Comparison,
};

pub const Options = struct {
    baseline_path: ?[]const u8 = null,
    candidate_path: ?[]const u8 = null,
    runner_filter: ?[]const u8 = null,
    warn_threshold: f64 = 0.05,
    fail_threshold: f64 = 0.10,
    json_output_path: ?[]const u8 = null,
    report_output_path: ?[]const u8 = null,
};

const TestRecordSpec = struct {
    benchmark_id: []const u8,
    runner: []const u8 = "zig",
    status: result.Status = .ok,
    mean_ns: ?f64 = 100.0,
    throughput_per_second: ?f64 = 10.0,
};

pub fn runCli(allocator: std.mem.Allocator) !void {
    const options = try parseArgs(allocator);
    const baseline_path = options.baseline_path orelse return error.MissingBaselinePath;
    const candidate_path = options.candidate_path orelse return error.MissingCandidatePath;
    if (options.fail_threshold < options.warn_threshold) return error.InvalidThresholdConfiguration;

    var baseline = try result.LoadedFile.loadFromFile(allocator, baseline_path);
    defer baseline.deinit();

    var candidate = try result.LoadedFile.loadFromFile(allocator, candidate_path);
    defer candidate.deinit();

    const report = try buildReport(
        allocator,
        baseline_path,
        candidate_path,
        baseline.records,
        candidate.records,
        options.runner_filter,
        .{
            .warn_ratio = options.warn_threshold,
            .fail_ratio = options.fail_threshold,
        },
    );

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try writeTextReport(stdout, report);
    try stdout.flush();

    if (options.report_output_path) |path| {
        try writeTextReportToPath(path, report);
    }
    if (options.json_output_path) |path| {
        try writeJsonReportToPath(path, report);
    }

    if (report.summary.should_fail) return error.BenchmarkRegressionDetected;
}

pub fn buildReport(
    allocator: std.mem.Allocator,
    baseline_path: []const u8,
    candidate_path: []const u8,
    baseline_records: []const result.Record,
    candidate_records: []const result.Record,
    runner_filter: ?[]const u8,
    thresholds: Thresholds,
) !Report {
    var comparisons = std.ArrayList(Comparison){};
    errdefer comparisons.deinit(allocator);

    const baseline_count = countFilteredRecords(baseline_records, runner_filter);
    const candidate_count = countFilteredRecords(candidate_records, runner_filter);

    var candidate_index = try buildIndex(allocator, candidate_records, runner_filter);
    defer candidate_index.deinit();

    const candidate_seen = try allocator.alloc(bool, candidate_records.len);
    @memset(candidate_seen, false);

    for (baseline_records) |baseline_record| {
        if (!runnerMatches(baseline_record, runner_filter)) continue;

        const key = try makeKey(allocator, baseline_record);
        const candidate_index_value = candidate_index.get(key);
        if (candidate_index_value) |candidate_idx| {
            candidate_seen[candidate_idx] = true;
            try comparisons.append(
                allocator,
                try comparePair(allocator, baseline_record, candidate_records[candidate_idx], thresholds),
            );
        } else {
            try comparisons.append(allocator, .{
                .benchmark_id = baseline_record.benchmark_id,
                .runner = baseline_record.runner,
                .classification = .missing_candidate,
                .baseline_status = baseline_record.status,
                .candidate_status = null,
                .baseline_mean_ns = if (baseline_record.stats) |stats| stats.mean_ns else null,
                .candidate_mean_ns = null,
                .baseline_throughput_per_second = if (baseline_record.stats) |stats| stats.throughput_per_second else null,
                .candidate_throughput_per_second = null,
                .notes = "benchmark record missing from candidate results",
            });
        }
    }

    for (candidate_records, 0..) |candidate_record, idx| {
        if (!runnerMatches(candidate_record, runner_filter)) continue;
        if (candidate_seen[idx]) continue;

        try comparisons.append(allocator, .{
            .benchmark_id = candidate_record.benchmark_id,
            .runner = candidate_record.runner,
            .classification = .new_candidate,
            .baseline_status = null,
            .candidate_status = candidate_record.status,
            .baseline_mean_ns = null,
            .candidate_mean_ns = if (candidate_record.stats) |stats| stats.mean_ns else null,
            .baseline_throughput_per_second = null,
            .candidate_throughput_per_second = if (candidate_record.stats) |stats| stats.throughput_per_second else null,
            .notes = "benchmark record only present in candidate results",
        });
    }

    const comparisons_slice = try comparisons.toOwnedSlice(allocator);
    return .{
        .baseline_path = baseline_path,
        .candidate_path = candidate_path,
        .runner_filter = runner_filter,
        .thresholds = thresholds,
        .summary = summarize(comparisons_slice, baseline_count, candidate_count),
        .comparisons = comparisons_slice,
    };
}

pub fn writeTextReport(writer: anytype, report: Report) !void {
    try writer.print(
        \\Benchmark comparison
        \\  baseline: {s}
        \\  candidate: {s}
        \\  warn threshold: {d:.2}%
        \\  fail threshold: {d:.2}%
        \\
    , .{
        report.baseline_path,
        report.candidate_path,
        report.thresholds.warn_ratio * 100.0,
        report.thresholds.fail_ratio * 100.0,
    });

    if (report.runner_filter) |runner| {
        try writer.print("  runner filter: {s}\n\n", .{runner});
    } else {
        try writer.writeAll("\n");
    }

    for (report.comparisons) |comparison| {
        try writer.print("{s} {s} [{s}]\n", .{
            classificationLabel(comparison.classification),
            comparison.benchmark_id,
            comparison.runner,
        });

        try writer.print("  status: {s} -> {s}\n", .{
            statusLabel(comparison.baseline_status),
            statusLabel(comparison.candidate_status),
        });

        if (comparison.baseline_mean_ns != null or comparison.candidate_mean_ns != null) {
            try writer.print("  mean latency (ns): ", .{});
            try writeOptionalFloat(writer, comparison.baseline_mean_ns);
            try writer.writeAll(" -> ");
            try writeOptionalFloat(writer, comparison.candidate_mean_ns);
            if (comparison.latency_delta_ratio) |delta| {
                try writer.print(" ({s}{d:.2}%)", .{ signPrefix(delta), @abs(delta) * 100.0 });
            }
            try writer.writeByte('\n');
        }

        if (comparison.baseline_throughput_per_second != null or comparison.candidate_throughput_per_second != null) {
            try writer.print("  throughput (/s): ", .{});
            try writeOptionalFloat(writer, comparison.baseline_throughput_per_second);
            try writer.writeAll(" -> ");
            try writeOptionalFloat(writer, comparison.candidate_throughput_per_second);
            if (comparison.throughput_delta_ratio) |delta| {
                try writer.print(" ({s}{d:.2}%)", .{ signPrefix(delta), @abs(delta) * 100.0 });
            }
            try writer.writeByte('\n');
        }

        if (comparison.notes) |notes| {
            try writer.print("  notes: {s}\n", .{notes});
        }

        try writer.writeByte('\n');
    }

    try writer.print(
        \\Summary
        \\  baseline records: {}
        \\  candidate records: {}
        \\  paired records: {}
        \\  improved: {}
        \\  passing: {}
        \\  warned: {}
        \\  failed: {}
        \\  skipped: {}
        \\  missing candidate: {}
        \\  new candidate: {}
        \\  should fail: {}
        \\
    , .{
        report.summary.baseline_records,
        report.summary.candidate_records,
        report.summary.paired_records,
        report.summary.improved,
        report.summary.passing,
        report.summary.warned,
        report.summary.failed,
        report.summary.skipped,
        report.summary.missing_candidate,
        report.summary.new_candidate,
        report.summary.should_fail,
    });
}

fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    var options = Options{};

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];

        if (std.mem.eql(u8, arg, "--baseline")) {
            options.baseline_path = try nextArg(args, &index, "--baseline");
        } else if (std.mem.eql(u8, arg, "--candidate")) {
            options.candidate_path = try nextArg(args, &index, "--candidate");
        } else if (std.mem.eql(u8, arg, "--runner")) {
            options.runner_filter = try nextArg(args, &index, "--runner");
        } else if (std.mem.eql(u8, arg, "--warn-threshold")) {
            const value = try nextArg(args, &index, "--warn-threshold");
            options.warn_threshold = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, arg, "--fail-threshold")) {
            const value = try nextArg(args, &index, "--fail-threshold");
            options.fail_threshold = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, arg, "--json-output")) {
            options.json_output_path = try nextArg(args, &index, "--json-output");
        } else if (std.mem.eql(u8, arg, "--report-output")) {
            options.report_output_path = try nextArg(args, &index, "--report-output");
        } else if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return error.HelpPrinted;
        } else {
            return error.UnknownArgument;
        }
    }

    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: benchmark-compare --baseline <path> --candidate <path> [--runner <name>] [--warn-threshold <fraction>] [--fail-threshold <fraction>] [--json-output <path>] [--report-output <path>]
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

fn writeTextReportToPath(path: []const u8, report: Report) !void {
    if (std.fs.path.dirname(path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try writeTextReport(writer, report);
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

fn buildIndex(
    allocator: std.mem.Allocator,
    records: []const result.Record,
    runner_filter: ?[]const u8,
) !std.StringHashMap(usize) {
    var index = std.StringHashMap(usize).init(allocator);
    errdefer index.deinit();

    try index.ensureTotalCapacity(@as(u32, @intCast(records.len)));
    for (records, 0..) |record_entry, idx| {
        if (!runnerMatches(record_entry, runner_filter)) continue;

        const key = try makeKey(allocator, record_entry);
        const gop = try index.getOrPut(key);
        if (gop.found_existing) {
            allocator.free(key);
            return error.DuplicateBenchmarkRecord;
        }
        gop.value_ptr.* = idx;
    }

    return index;
}

fn comparePair(
    allocator: std.mem.Allocator,
    baseline: result.Record,
    candidate: result.Record,
    thresholds: Thresholds,
) !Comparison {
    var comparison = Comparison{
        .benchmark_id = baseline.benchmark_id,
        .runner = baseline.runner,
        .classification = .pass,
        .baseline_status = baseline.status,
        .candidate_status = candidate.status,
        .baseline_mean_ns = if (baseline.stats) |stats| stats.mean_ns else null,
        .candidate_mean_ns = if (candidate.stats) |stats| stats.mean_ns else null,
        .baseline_throughput_per_second = if (baseline.stats) |stats| stats.throughput_per_second else null,
        .candidate_throughput_per_second = if (candidate.stats) |stats| stats.throughput_per_second else null,
    };

    if (!std.mem.eql(u8, baseline.suite, candidate.suite) or
        !std.mem.eql(u8, baseline.kind, candidate.kind) or
        !std.mem.eql(u8, baseline.dtype, candidate.dtype))
    {
        comparison.classification = .fail;
        comparison.notes = "matching benchmark ids differ in suite, kind, or dtype";
        return comparison;
    }

    if (baseline.status == .ok and candidate.status == .ok) {
        if (baseline.stats == null or candidate.stats == null) {
            comparison.classification = .fail;
            comparison.notes = "ok records must contain summary statistics";
            return comparison;
        }

        comparison.latency_delta_ratio = try percentageDelta(baseline.stats.?.mean_ns, candidate.stats.?.mean_ns);

        if (baseline.stats.?.throughput_per_second) |baseline_throughput| {
            if (candidate.stats.?.throughput_per_second) |candidate_throughput| {
                const baseline_unit = baseline.stats.?.throughput_unit;
                const candidate_unit = candidate.stats.?.throughput_unit;
                if (optionalStringsEqual(baseline_unit, candidate_unit)) {
                    comparison.throughput_delta_ratio = try percentageDelta(baseline_throughput, candidate_throughput);
                } else {
                    comparison.notes = "throughput units differ between baseline and candidate";
                }
            }
        }

        const latency_delta = comparison.latency_delta_ratio orelse 0.0;
        comparison.classification = if (latency_delta > thresholds.fail_ratio)
            .fail
        else if (latency_delta > thresholds.warn_ratio)
            .warn
        else if (latency_delta < 0.0)
            .improved
        else
            .pass;

        return comparison;
    }

    if (baseline.status == .ok and candidate.status != .ok) {
        comparison.classification = .fail;
        comparison.notes = try std.fmt.allocPrint(
            allocator,
            "candidate status regressed from ok to {s}",
            .{@tagName(candidate.status)},
        );
        return comparison;
    }

    if (baseline.status != .ok and candidate.status == .ok) {
        comparison.classification = .improved;
        comparison.notes = try std.fmt.allocPrint(
            allocator,
            "candidate status improved from {s} to ok",
            .{@tagName(baseline.status)},
        );
        return comparison;
    }

    comparison.classification = .skipped;
    comparison.notes = try std.fmt.allocPrint(
        allocator,
        "neither result was comparable ({s} -> {s})",
        .{ @tagName(baseline.status), @tagName(candidate.status) },
    );
    return comparison;
}

fn countFilteredRecords(records: []const result.Record, runner_filter: ?[]const u8) usize {
    var count: usize = 0;
    for (records) |record_entry| {
        if (runnerMatches(record_entry, runner_filter)) count += 1;
    }
    return count;
}

fn summarize(
    comparisons: []const Comparison,
    baseline_records: usize,
    candidate_records: usize,
) Summary {
    var summary = Summary{
        .baseline_records = baseline_records,
        .candidate_records = candidate_records,
    };

    for (comparisons) |comparison| {
        switch (comparison.classification) {
            .improved => {
                summary.paired_records += 1;
                summary.improved += 1;
            },
            .pass => {
                summary.paired_records += 1;
                summary.passing += 1;
            },
            .warn => {
                summary.paired_records += 1;
                summary.warned += 1;
            },
            .fail => {
                summary.paired_records += 1;
                summary.failed += 1;
                summary.should_fail = true;
            },
            .skipped => {
                summary.paired_records += 1;
                summary.skipped += 1;
            },
            .missing_candidate => {
                summary.missing_candidate += 1;
                summary.should_fail = true;
            },
            .new_candidate => summary.new_candidate += 1,
        }
    }

    return summary;
}

fn makeKey(allocator: std.mem.Allocator, record_entry: result.Record) ![]const u8 {
    return std.fmt.allocPrint(allocator, "{s}\x1f{s}", .{ record_entry.benchmark_id, record_entry.runner });
}

fn optionalStringsEqual(lhs: ?[]const u8, rhs: ?[]const u8) bool {
    if (lhs == null and rhs == null) return true;
    if (lhs == null or rhs == null) return false;
    return std.mem.eql(u8, lhs.?, rhs.?);
}

fn percentageDelta(baseline: f64, candidate: f64) !f64 {
    if (baseline <= 0.0) return error.InvalidBaselineMetric;
    return (candidate - baseline) / baseline;
}

fn runnerMatches(record_entry: result.Record, runner_filter: ?[]const u8) bool {
    if (runner_filter) |runner| {
        return std.mem.eql(u8, record_entry.runner, runner);
    }
    return true;
}

fn classificationLabel(classification: Classification) []const u8 {
    return switch (classification) {
        .improved => "IMPROVED",
        .pass => "PASS",
        .warn => "WARN",
        .fail => "FAIL",
        .skipped => "SKIPPED",
        .missing_candidate => "MISSING",
        .new_candidate => "NEW",
    };
}

fn statusLabel(status: ?result.Status) []const u8 {
    return if (status) |value| @tagName(value) else "missing";
}

fn signPrefix(value: f64) []const u8 {
    return if (value >= 0.0) "+" else "-";
}

fn writeOptionalFloat(writer: anytype, value: ?f64) !void {
    if (value) |number| {
        try writer.print("{d:.3}", .{number});
    } else {
        try writer.writeAll("n/a");
    }
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

    try builder.writer(allocator).print(
        "{{\"benchmark_id\":\"{s}\",\"suite\":\"primitive\",\"kind\":\"primitive_add\",\"runner\":\"{s}\",\"status\":\"{s}\",\"dtype\":\"f32\",\"warmup_iterations\":1,\"measured_iterations\":2,\"batch_size\":null,\"seed\":1,\"shapes\":[{{\"name\":\"lhs\",\"dims\":[1]}}],\"runtime\":{{\"timestamp_unix_ms\":0,\"git_commit\":\"deadbeef\",\"git_dirty\":false,\"zig_version\":\"0.15.2\",\"harness_version\":\"0.1.0\"}},\"system\":{{\"os\":\"linux\",\"kernel\":\"test\",\"arch\":\"x86_64\",\"cpu_model\":\"cpu\",\"cpu_logical_cores\":1,\"total_memory_bytes\":null}},\"backend\":{{\"device\":\"host\",\"host_provider\":\"blas\",\"thread_count\":1,\"accelerator\":null}},\"setup_latency_ns\":10,\"stats\":{s},\"notes\":null}}\n",
        .{
            spec.benchmark_id,
            spec.runner,
            @tagName(spec.status),
            stats_json,
        },
    );
}

fn loadTestRecords(
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

test "comparison thresholds classify improvements warnings and failures" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var baseline = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.pass", .mean_ns = 100.0, .throughput_per_second = 20.0 },
        .{ .benchmark_id = "bench.warn", .mean_ns = 100.0, .throughput_per_second = 20.0 },
        .{ .benchmark_id = "bench.fail", .mean_ns = 100.0, .throughput_per_second = 20.0 },
        .{ .benchmark_id = "bench.improve", .mean_ns = 100.0, .throughput_per_second = 20.0 },
    });
    defer baseline.deinit();

    var candidate = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.pass", .mean_ns = 104.0, .throughput_per_second = 19.0 },
        .{ .benchmark_id = "bench.warn", .mean_ns = 107.0, .throughput_per_second = 18.0 },
        .{ .benchmark_id = "bench.fail", .mean_ns = 112.0, .throughput_per_second = 17.0 },
        .{ .benchmark_id = "bench.improve", .mean_ns = 92.0, .throughput_per_second = 22.0 },
    });
    defer candidate.deinit();

    const report = try buildReport(
        allocator,
        "baseline.jsonl",
        "candidate.jsonl",
        baseline.records,
        candidate.records,
        null,
        .{},
    );

    try std.testing.expectEqual(@as(usize, 4), report.comparisons.len);
    try std.testing.expectEqual(Classification.pass, report.comparisons[0].classification);
    try std.testing.expectEqual(Classification.warn, report.comparisons[1].classification);
    try std.testing.expectEqual(Classification.fail, report.comparisons[2].classification);
    try std.testing.expectEqual(Classification.improved, report.comparisons[3].classification);
    try std.testing.expect(report.summary.should_fail);
    try std.testing.expectEqual(@as(usize, 1), report.summary.warned);
    try std.testing.expectEqual(@as(usize, 1), report.summary.failed);
    try std.testing.expectEqual(@as(usize, 1), report.summary.improved);
}

test "comparison handles missing new and non-ok records" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var baseline = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.missing", .status = .ok, .mean_ns = 100.0 },
        .{ .benchmark_id = "bench.regressed-status", .status = .ok, .mean_ns = 100.0 },
        .{ .benchmark_id = "bench.improved-status", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
        .{ .benchmark_id = "bench.skipped-both", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
    });
    defer baseline.deinit();

    var candidate = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.regressed-status", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
        .{ .benchmark_id = "bench.improved-status", .status = .ok, .mean_ns = 90.0, .throughput_per_second = 11.0 },
        .{ .benchmark_id = "bench.skipped-both", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
        .{ .benchmark_id = "bench.new", .status = .ok, .mean_ns = 80.0, .throughput_per_second = 12.0 },
    });
    defer candidate.deinit();

    const report = try buildReport(
        allocator,
        "baseline.jsonl",
        "candidate.jsonl",
        baseline.records,
        candidate.records,
        null,
        .{},
    );

    try std.testing.expectEqual(@as(usize, 5), report.comparisons.len);
    try std.testing.expectEqual(@as(usize, 1), report.summary.missing_candidate);
    try std.testing.expectEqual(@as(usize, 1), report.summary.new_candidate);
    try std.testing.expectEqual(@as(usize, 1), report.summary.improved);
    try std.testing.expectEqual(@as(usize, 1), report.summary.failed);
    try std.testing.expectEqual(@as(usize, 1), report.summary.skipped);
    try std.testing.expect(report.summary.should_fail);
}

test "comparison can filter by runner" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var baseline = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.host", .runner = "zig", .mean_ns = 100.0 },
        .{ .benchmark_id = "bench.host", .runner = "pytorch", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
    });
    defer baseline.deinit();

    var candidate = try loadTestRecords(allocator, &.{
        .{ .benchmark_id = "bench.host", .runner = "zig", .mean_ns = 101.0 },
        .{ .benchmark_id = "bench.host", .runner = "pytorch", .status = .skipped, .mean_ns = null, .throughput_per_second = null },
    });
    defer candidate.deinit();

    const report = try buildReport(
        allocator,
        "baseline.jsonl",
        "candidate.jsonl",
        baseline.records,
        candidate.records,
        "zig",
        .{},
    );

    try std.testing.expectEqual(@as(usize, 1), report.comparisons.len);
    try std.testing.expectEqualStrings("zig", report.comparisons[0].runner);
    try std.testing.expectEqual(@as(usize, 1), report.summary.baseline_records);
    try std.testing.expectEqual(@as(usize, 1), report.summary.candidate_records);
}
