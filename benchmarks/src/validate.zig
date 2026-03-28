const std = @import("std");
const cli = @import("cli.zig");
const manifest = @import("manifest.zig");
const result = @import("result.zig");
const workload = @import("workload.zig");

pub const Scope = enum {
    spec,
    result,
};

pub const Issue = struct {
    scope: Scope,
    path: []const u8,
    benchmark_id: ?[]const u8 = null,
    runner: ?[]const u8 = null,
    thread_count: ?u32 = null,
    message: []const u8,
};

pub const Summary = struct {
    specs_checked: usize = 0,
    result_files_checked: usize = 0,
    records_checked: usize = 0,
    issues: usize = 0,
    should_fail: bool = false,
};

pub const Report = struct {
    summary: Summary,
    issues: []const Issue,
};

pub const Options = struct {
    spec_root: []const u8 = "benchmarks/specs",
    group: []const u8 = "all",
    spec_path: ?[]const u8 = null,
    input_paths: []const []const u8 = &.{},
    json_output_path: ?[]const u8 = null,
    report_output_path: ?[]const u8 = null,
};

const SpecCache = struct {
    allocator: std.mem.Allocator,
    map: std.StringHashMap(manifest.Spec),

    fn init(allocator: std.mem.Allocator) SpecCache {
        return .{
            .allocator = allocator,
            .map = std.StringHashMap(manifest.Spec).init(allocator),
        };
    }

    fn deinit(self: *SpecCache) void {
        self.map.deinit();
    }

    fn getOrLoad(self: *SpecCache, path: []const u8) !manifest.Spec {
        if (self.map.get(path)) |spec| return spec;

        const owned_path = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(owned_path);

        const spec = try manifest.loadFromFile(self.allocator, path);
        const gop = try self.map.getOrPut(owned_path);
        if (gop.found_existing) {
            self.allocator.free(owned_path);
            return gop.value_ptr.*;
        }
        gop.value_ptr.* = spec;
        return spec;
    }
};

pub fn runCli(allocator: std.mem.Allocator) !void {
    const options = try parseArgs(allocator);
    const report = try buildReport(allocator, options);

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

    if (report.summary.should_fail) return error.BenchmarkValidationFailed;
}

pub fn buildReport(allocator: std.mem.Allocator, options: Options) !Report {
    var issues = std.ArrayList(Issue){};
    errdefer issues.deinit(allocator);

    var summary = Summary{};

    if (shouldValidateSelectedSpecs(options)) {
        try validateSelectedSpecs(allocator, options, &summary, &issues);
    }

    var spec_cache = SpecCache.init(allocator);
    defer spec_cache.deinit();

    for (options.input_paths) |input_path| {
        try validateResultFile(allocator, input_path, &spec_cache, &summary, &issues);
    }

    summary.issues = issues.items.len;
    summary.should_fail = summary.issues != 0;

    return .{
        .summary = summary,
        .issues = try issues.toOwnedSlice(allocator),
    };
}

pub fn writeTextReport(writer: anytype, report: Report) !void {
    try writer.writeAll("Benchmark validation\n");

    if (report.issues.len == 0) {
        try writer.writeAll("  no validation issues detected\n\n");
    } else {
        for (report.issues) |issue| {
            try writer.print("[{s}] {s}", .{ @tagName(issue.scope), issue.path });
            if (issue.benchmark_id) |benchmark_id| {
                try writer.print(" :: {s}", .{benchmark_id});
            }
            if (issue.runner) |runner| {
                try writer.print(" [{s}", .{runner});
                if (issue.thread_count) |thread_count| {
                    try writer.print(", threads={d}", .{thread_count});
                }
                try writer.writeByte(']');
            } else if (issue.thread_count) |thread_count| {
                try writer.print(" [threads={d}]", .{thread_count});
            }
            try writer.print(": {s}\n", .{issue.message});
        }
        try writer.writeByte('\n');
    }

    try writer.print(
        \\Summary
        \\  spec files checked: {}
        \\  result files checked: {}
        \\  result records checked: {}
        \\  issues: {}
        \\  should fail: {}
        \\
    , .{
        report.summary.specs_checked,
        report.summary.result_files_checked,
        report.summary.records_checked,
        report.summary.issues,
        report.summary.should_fail,
    });
}

fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    var input_paths = std.ArrayList([]const u8){};
    errdefer input_paths.deinit(allocator);

    var options = Options{};

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];

        if (std.mem.eql(u8, arg, "--spec-root")) {
            options.spec_root = try nextArg(args, &index, "--spec-root");
        } else if (std.mem.eql(u8, arg, "--group")) {
            options.group = try nextArg(args, &index, "--group");
        } else if (std.mem.eql(u8, arg, "--spec")) {
            options.spec_path = try nextArg(args, &index, "--spec");
        } else if (std.mem.eql(u8, arg, "--input")) {
            try input_paths.append(allocator, try nextArg(args, &index, "--input"));
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

    options.input_paths = try input_paths.toOwnedSlice(allocator);
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: benchmark-validate [--spec-root <path>] [--group primitive|blas|autograd|memory|model-train|model-infer|models|all] [--spec <path>] [--input <path> ...] [--json-output <path>] [--report-output <path>]
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

fn shouldValidateSelectedSpecs(options: Options) bool {
    return options.input_paths.len == 0 or
        options.spec_path != null or
        !std.mem.eql(u8, options.group, "all") or
        !std.mem.eql(u8, options.spec_root, "benchmarks/specs");
}

fn validateSelectedSpecs(
    allocator: std.mem.Allocator,
    options: Options,
    summary: *Summary,
    issues: *std.ArrayList(Issue),
) !void {
    const selection = cli.Options{
        .spec_root = options.spec_root,
        .group = options.group,
        .spec_path = options.spec_path,
    };
    const specs = cli.loadSpecs(allocator, selection) catch |err| {
        try appendIssueFmt(allocator, issues, .spec, options.spec_path orelse options.spec_root, null, null, null, "failed to load selected benchmark specs: {s}", .{@errorName(err)});
        return;
    };

    var ids = std.StringHashMap([]const u8).init(allocator);
    defer ids.deinit();

    for (specs) |spec| {
        summary.specs_checked += 1;

        const gop = try ids.getOrPut(spec.id);
        if (gop.found_existing) {
            try appendIssueFmt(
                allocator,
                issues,
                .spec,
                spec.path,
                spec.id,
                null,
                spec.thread_count,
                "duplicate benchmark id also declared in {s}",
                .{gop.value_ptr.*},
            );
        } else {
            gop.value_ptr.* = spec.path;
        }

        if (!suitePathMatches(spec.path, spec.suite.asString())) {
            try appendIssueFmt(
                allocator,
                issues,
                .spec,
                spec.path,
                spec.id,
                null,
                spec.thread_count,
                "spec path does not match declared suite `{s}`",
                .{spec.suite.asString()},
            );
        }

        if (spec.pytorch_runner) |runner| {
            std.fs.cwd().access(runner, .{}) catch {
                try appendIssueFmt(
                    allocator,
                    issues,
                    .spec,
                    spec.path,
                    spec.id,
                    null,
                    spec.thread_count,
                    "declared PyTorch runner `{s}` does not exist",
                    .{runner},
                );
            };
        }
    }
}

fn validateResultFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    spec_cache: *SpecCache,
    summary: *Summary,
    issues: *std.ArrayList(Issue),
) !void {
    summary.result_files_checked += 1;

    var loaded = result.LoadedFile.loadFromFile(allocator, path) catch |err| {
        try appendIssueFmt(allocator, issues, .result, path, null, null, null, "failed to load JSONL records: {s}", .{@errorName(err)});
        return;
    };
    defer loaded.deinit();

    var identities = std.StringHashMap(void).init(allocator);
    defer identities.deinit();

    for (loaded.records) |record_entry| {
        summary.records_checked += 1;

        const identity = try identityKey(allocator, record_entry);
        const gop = try identities.getOrPut(identity);
        if (gop.found_existing) {
            try appendIssueFmt(
                allocator,
                issues,
                .result,
                path,
                record_entry.benchmark_id,
                record_entry.runner,
                record_entry.backend.thread_count,
                "duplicate record identity within result file",
                .{},
            );
        }

        try validateRecordContract(allocator, path, record_entry, issues);
        try validateRecordAgainstSpec(allocator, path, record_entry, spec_cache, issues);
    }
}

fn validateRecordContract(
    allocator: std.mem.Allocator,
    path: []const u8,
    record_entry: result.Record,
    issues: *std.ArrayList(Issue),
) !void {
    if (trimmedLen(record_entry.benchmark_id) == 0) {
        try appendIssueFmt(allocator, issues, .result, path, null, record_entry.runner, record_entry.backend.thread_count, "benchmark id is empty", .{});
    }
    if (trimmedLen(record_entry.suite) == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "suite is empty", .{});
    }
    if (trimmedLen(record_entry.kind) == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "kind is empty", .{});
    }
    if (trimmedLen(record_entry.dtype) == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "dtype is empty", .{});
    }
    if (trimmedLen(record_entry.runner) == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, null, record_entry.backend.thread_count, "runner is empty", .{});
    }
    if (record_entry.measured_iterations == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "measured_iterations must be greater than zero", .{});
    }
    if (record_entry.shapes.len == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "record does not include shape metadata", .{});
    }

    if (record_entry.provenance) |provenance| {
        if (trimmedLen(provenance.data_source) == 0) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "provenance data source is empty", .{});
        }
        if (provenance.preprocessing.len == 0) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "provenance preprocessing is empty", .{});
        }
        for (provenance.preprocessing) |step| {
            if (trimmedLen(step) == 0) {
                try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "provenance preprocessing contains an empty step", .{});
                break;
            }
        }
    } else {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "record is missing provenance metadata", .{});
    }

    if (record_entry.runtime.timestamp_unix_ms <= 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "runtime timestamp is missing or invalid", .{});
    }
    inline for (.{
        .{ "runtime git commit", record_entry.runtime.git_commit },
        .{ "runtime zig version", record_entry.runtime.zig_version },
        .{ "runtime harness version", record_entry.runtime.harness_version },
        .{ "system os", record_entry.system.os },
        .{ "system kernel", record_entry.system.kernel },
        .{ "system arch", record_entry.system.arch },
        .{ "system cpu model", record_entry.system.cpu_model },
        .{ "backend device", record_entry.backend.device },
        .{ "backend host provider", record_entry.backend.host_provider },
    }) |field| {
        if (trimmedLen(field[1]) == 0) {
            try appendIssueFmt(
                allocator,
                issues,
                .result,
                path,
                record_entry.benchmark_id,
                record_entry.runner,
                record_entry.backend.thread_count,
                "{s} is empty",
                .{field[0]},
            );
        }
    }
    if (record_entry.system.cpu_logical_cores == 0) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "system cpu_logical_cores must be greater than zero", .{});
    }
    if (std.mem.eql(u8, record_entry.backend.device, "cuda")) {
        if (record_entry.backend.accelerator == null or trimmedLen(record_entry.backend.accelerator.?) == 0) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "cuda records must include an accelerator label", .{});
        }
        if (record_entry.status == .ok and record_entry.backend.cuda == null) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "ok cuda records must include CUDA device metadata", .{});
        }
        if (record_entry.backend.cuda) |cuda| {
            if (trimmedLen(cuda.device_name) == 0) {
                try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "cuda device_name is empty", .{});
            }
            if (trimmedLen(cuda.driver_version) == 0 or trimmedLen(cuda.runtime_version) == 0) {
                try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "cuda version metadata is empty", .{});
            }
        }
    } else if (record_entry.backend.cuda != null) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "non-cuda records must not include CUDA device metadata", .{});
    }

    if (record_entry.status == .ok) {
        if (record_entry.stats == null) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "ok records must include summary statistics", .{});
        }
        if (record_entry.setup_latency_ns == null) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "ok records must include setup latency", .{});
        }
    }

    if (record_entry.stats) |stats| {
        try validateStats(allocator, path, record_entry, stats, issues);
    }
}

fn validateStats(
    allocator: std.mem.Allocator,
    path: []const u8,
    record_entry: result.Record,
    stats: result.SummaryStats,
    issues: *std.ArrayList(Issue),
) !void {
    if (stats.min_ns > stats.median_ns or
        stats.median_ns > stats.p95_ns or
        stats.p95_ns > stats.max_ns)
    {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "summary stats are not monotonically ordered", .{});
    }
    if (!std.math.isFinite(stats.mean_ns) or stats.mean_ns < @as(f64, @floatFromInt(stats.min_ns)) or stats.mean_ns > @as(f64, @floatFromInt(stats.max_ns))) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "summary mean_ns is not finite or falls outside min/max bounds", .{});
    }
    if (stats.throughput_per_second) |throughput| {
        if (!std.math.isFinite(throughput) or throughput <= 0) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "throughput_per_second must be finite and positive", .{});
        }
        if (stats.throughput_unit == null or trimmedLen(stats.throughput_unit.?) == 0) {
            try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "throughput_unit must be present when throughput_per_second is set", .{});
        }
    } else if (stats.throughput_unit != null) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "throughput_unit is set without throughput_per_second", .{});
    }
}

fn validateRecordAgainstSpec(
    allocator: std.mem.Allocator,
    path: []const u8,
    record_entry: result.Record,
    spec_cache: *SpecCache,
    issues: *std.ArrayList(Issue),
) !void {
    const spec_path = record_entry.spec_path orelse {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "record is missing spec_path", .{});
        return;
    };

    const spec = spec_cache.getOrLoad(spec_path) catch |err| {
        try appendIssueFmt(
            allocator,
            issues,
            .result,
            path,
            record_entry.benchmark_id,
            record_entry.runner,
            record_entry.backend.thread_count,
            "failed to load referenced spec `{s}`: {s}",
            .{ spec_path, @errorName(err) },
        );
        return;
    };

    inline for (.{
        .{ "benchmark id", record_entry.benchmark_id, spec.id },
        .{ "suite", record_entry.suite, spec.suite.asString() },
        .{ "kind", record_entry.kind, spec.kind.asString() },
        .{ "dtype", record_entry.dtype, spec.dtype.asString() },
    }) |field| {
        if (!std.mem.eql(u8, field[1], field[2])) {
            try appendIssueFmt(
                allocator,
                issues,
                .result,
                path,
                record_entry.benchmark_id,
                record_entry.runner,
                record_entry.backend.thread_count,
                "{s} does not match referenced spec (`{s}` vs `{s}`)",
                .{ field[0], field[1], field[2] },
            );
        }
    }

    if (record_entry.warmup_iterations != spec.warmup_iterations) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "warmup_iterations does not match referenced spec", .{});
    }
    if (record_entry.measured_iterations != spec.measured_iterations) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "measured_iterations does not match referenced spec", .{});
    }
    if (record_entry.seed != spec.seed) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "seed does not match referenced spec", .{});
    }
    if (!provenanceMatches(record_entry.provenance, spec.provenance)) {
        try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "record provenance does not match referenced spec", .{});
    }
    if (std.mem.eql(u8, record_entry.runner, "zig")) {
        const expected_device = spec.device.kind.asString();
        if (!std.mem.eql(u8, record_entry.backend.device, expected_device)) {
            try appendIssueFmt(
                allocator,
                issues,
                .result,
                path,
                record_entry.benchmark_id,
                record_entry.runner,
                record_entry.backend.thread_count,
                "backend device does not match referenced spec (`{s}` vs `{s}`)",
                .{ record_entry.backend.device, expected_device },
            );
        }

        switch (spec.device.kind) {
            .host => {
                if (record_entry.backend.accelerator != null) {
                    try appendIssueFmt(allocator, issues, .result, path, record_entry.benchmark_id, record_entry.runner, record_entry.backend.thread_count, "host zig records must not include accelerator labels", .{});
                }
            },
            .cuda => {
                const expected_accelerator = try std.fmt.allocPrint(
                    allocator,
                    "cuda:{d}",
                    .{spec.device.cuda_device_index},
                );
                if (record_entry.backend.accelerator == null or
                    !std.mem.eql(u8, record_entry.backend.accelerator.?, expected_accelerator))
                {
                    try appendIssueFmt(
                        allocator,
                        issues,
                        .result,
                        path,
                        record_entry.benchmark_id,
                        record_entry.runner,
                        record_entry.backend.thread_count,
                        "backend accelerator does not match referenced spec (`{s}` vs `{s}`)",
                        .{ record_entry.backend.accelerator orelse "none", expected_accelerator },
                    );
                }
            },
        }
    }

    const expected_batch_size = workload.expectedBatchSize(spec);
    if (record_entry.batch_size != expected_batch_size) {
        try appendIssueFmt(
            allocator,
            issues,
            .result,
            path,
            record_entry.benchmark_id,
            record_entry.runner,
            record_entry.backend.thread_count,
            "batch_size does not match workload expectations for `{s}`",
            .{spec.kind.asString()},
        );
    }

    const expected_shapes = try workload.expectedShapeMetadata(allocator, spec);
    if (!shapeMetadataMatches(record_entry.shapes, expected_shapes)) {
        try appendIssueFmt(
            allocator,
            issues,
            .result,
            path,
            record_entry.benchmark_id,
            record_entry.runner,
            record_entry.backend.thread_count,
            "shape metadata does not match referenced spec",
            .{},
        );
    }
}

fn appendIssueFmt(
    allocator: std.mem.Allocator,
    issues: *std.ArrayList(Issue),
    scope: Scope,
    path: []const u8,
    benchmark_id: ?[]const u8,
    runner: ?[]const u8,
    thread_count: ?u32,
    comptime fmt: []const u8,
    args: anytype,
) !void {
    try issues.append(allocator, .{
        .scope = scope,
        .path = path,
        .benchmark_id = benchmark_id,
        .runner = runner,
        .thread_count = thread_count,
        .message = try std.fmt.allocPrint(allocator, fmt, args),
    });
}

fn identityKey(allocator: std.mem.Allocator, record_entry: result.Record) ![]const u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}\x1f{s}\x1f{s}\x1f{s}\x1f{s}\x1f{s}",
        .{
            record_entry.benchmark_id,
            record_entry.runner,
            record_entry.backend.device,
            record_entry.backend.host_provider,
            record_entry.backend.accelerator orelse "none",
            if (record_entry.backend.thread_count) |thread_count|
                try std.fmt.allocPrint(allocator, "{d}", .{thread_count})
            else
                "none",
        },
    );
}

fn suitePathMatches(path: []const u8, suite: []const u8) bool {
    const dir_name = std.fs.path.dirname(path) orelse return false;
    return std.mem.eql(u8, std.fs.path.basename(dir_name), suite);
}

fn trimmedLen(value: []const u8) usize {
    return std.mem.trim(u8, value, " \t\r\n").len;
}

fn provenanceMatches(actual: ?result.BenchmarkProvenance, expected: result.BenchmarkProvenance) bool {
    if (actual == null) return false;
    if (!std.mem.eql(u8, actual.?.data_source, expected.data_source)) return false;
    if (actual.?.preprocessing.len != expected.preprocessing.len) return false;
    for (actual.?.preprocessing, expected.preprocessing) |actual_step, expected_step| {
        if (!std.mem.eql(u8, actual_step, expected_step)) return false;
    }
    return true;
}

fn shapeMetadataMatches(
    actual: []const result.ShapeMetadata,
    expected: []const result.ShapeMetadata,
) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |actual_shape, expected_shape| {
        if (!std.mem.eql(u8, actual_shape.name, expected_shape.name)) return false;
        if (!std.mem.eql(usize, actual_shape.dims, expected_shape.dims)) return false;
    }
    return true;
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

test "validator accepts matching benchmark record and spec" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const spec_rel = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/specs/primitive/add.json", .{tmp.sub_path[0..]});
    const result_rel = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/results/primitive.jsonl", .{tmp.sub_path[0..]});

    try tmp.dir.makePath("specs/primitive");
    try tmp.dir.makePath("results");
    try tmp.dir.writeFile(.{
        .sub_path = "specs/primitive/add.json",
        .data =
        \\{
        \\  "id": "primitive.add.synthetic",
        \\  "suite": "primitive",
        \\  "kind": "primitive_add",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "seed": 7,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape lhs", "reshape rhs"]
        \\  },
        \\  "lhs_shape": [4, 4],
        \\  "rhs_shape": [4, 4]
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "results/primitive.jsonl",
        .data = try std.fmt.allocPrint(
            allocator,
            "{{\"benchmark_id\":\"primitive.add.synthetic\",\"spec_path\":\"{s}\",\"suite\":\"primitive\",\"kind\":\"primitive_add\",\"runner\":\"zig\",\"status\":\"ok\",\"dtype\":\"f32\",\"warmup_iterations\":1,\"measured_iterations\":2,\"batch_size\":null,\"seed\":7,\"shapes\":[{{\"name\":\"lhs\",\"dims\":[4,4]}},{{\"name\":\"rhs\",\"dims\":[4,4]}}],\"provenance\":{{\"data_source\":\"synthetic.splitmix64\",\"preprocessing\":[\"reshape lhs\",\"reshape rhs\"]}},\"runtime\":{{\"timestamp_unix_ms\":1,\"git_commit\":\"deadbeef\",\"git_dirty\":false,\"zig_version\":\"0.15.2\",\"harness_version\":\"0.1.0\"}},\"system\":{{\"os\":\"macos\",\"kernel\":\"Darwin 24.0\",\"arch\":\"aarch64\",\"cpu_model\":\"cpu\",\"cpu_logical_cores\":8,\"cpu_frequency_policy\":null,\"total_memory_bytes\":null}},\"backend\":{{\"device\":\"host\",\"host_provider\":\"accelerate\",\"thread_count\":1,\"accelerator\":null,\"thread_environment\":{{\"omp_num_threads\":\"1\"}}}},\"setup_latency_ns\":10,\"stats\":{{\"min_ns\":10,\"median_ns\":12,\"mean_ns\":12.0,\"p95_ns\":13,\"max_ns\":13,\"throughput_per_second\":100.0,\"throughput_unit\":\"elements\"}},\"memory\":null,\"notes\":null}}\n",
            .{spec_rel},
        ),
    });

    const input_paths = try allocator.alloc([]const u8, 1);
    input_paths[0] = result_rel;

    const report = try buildReport(allocator, .{
        .input_paths = input_paths,
    });

    try std.testing.expectEqual(@as(usize, 0), report.summary.issues);
    try std.testing.expectEqual(@as(usize, 1), report.summary.records_checked);
}

test "validator flags duplicate spec ids and missing runner paths" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const spec_root = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/specs", .{tmp.sub_path[0..]});

    try tmp.dir.makePath("specs/primitive");
    try tmp.dir.makePath("specs/blas");
    try tmp.dir.writeFile(.{
        .sub_path = "specs/primitive/first.json",
        .data =
        \\{
        \\  "id": "duplicate.id",
        \\  "suite": "primitive",
        \\  "kind": "primitive_add",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "seed": 7,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape lhs", "reshape rhs"]
        \\  },
        \\  "lhs_shape": [4, 4],
        \\  "rhs_shape": [4, 4],
        \\  "pytorch_runner": "benchmarks/runners/pytorch/does-not-exist.py"
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "specs/blas/second.json",
        .data =
        \\{
        \\  "id": "duplicate.id",
        \\  "suite": "blas",
        \\  "kind": "blas_dot",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "seed": 7,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape lhs", "reshape rhs"]
        \\  },
        \\  "lhs_shape": [16],
        \\  "rhs_shape": [16]
        \\}
        ,
    });

    const report = try buildReport(allocator, .{
        .spec_root = spec_root,
    });

    try std.testing.expect(report.summary.should_fail);
    try std.testing.expect(report.summary.issues >= 2);
}

test "validator flags spec mismatches and duplicate result identities" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const spec_rel = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/specs/model-infer/mnist.json", .{tmp.sub_path[0..]});
    const result_rel = try std.fmt.allocPrint(allocator, ".zig-cache/tmp/{s}/results/models.jsonl", .{tmp.sub_path[0..]});

    try tmp.dir.makePath("specs/model-infer");
    try tmp.dir.makePath("results");
    try tmp.dir.writeFile(.{
        .sub_path = "specs/model-infer/mnist.json",
        .data =
        \\{
        \\  "id": "model-infer.mnist.synthetic",
        \\  "suite": "model-infer",
        \\  "kind": "mnist_mlp_infer",
        \\  "dtype": "f32",
        \\  "warmup_iterations": 1,
        \\  "measured_iterations": 2,
        \\  "batch_size": 8,
        \\  "thread_count": 1,
        \\  "seed": 7,
        \\  "provenance": {
        \\    "data_source": "synthetic.splitmix64",
        \\    "preprocessing": ["reshape inputs to input_shape"]
        \\  },
        \\  "input_shape": [8, 1, 28, 28]
        \\}
        ,
    });
    try tmp.dir.writeFile(.{
        .sub_path = "results/models.jsonl",
        .data = try std.fmt.allocPrint(
            allocator,
            "{{\"benchmark_id\":\"model-infer.mnist.synthetic\",\"spec_path\":\"{s}\",\"suite\":\"model-infer\",\"kind\":\"mnist_mlp_infer\",\"runner\":\"zig\",\"status\":\"ok\",\"dtype\":\"f32\",\"warmup_iterations\":1,\"measured_iterations\":2,\"batch_size\":8,\"seed\":7,\"shapes\":[{{\"name\":\"input\",\"dims\":[8,1,28,28]}}],\"provenance\":{{\"data_source\":\"synthetic.splitmix64\",\"preprocessing\":[\"reshape inputs to input_shape\"]}},\"runtime\":{{\"timestamp_unix_ms\":1,\"git_commit\":\"deadbeef\",\"git_dirty\":false,\"zig_version\":\"0.15.2\",\"harness_version\":\"0.1.0\"}},\"system\":{{\"os\":\"macos\",\"kernel\":\"Darwin 24.0\",\"arch\":\"aarch64\",\"cpu_model\":\"cpu\",\"cpu_logical_cores\":8,\"cpu_frequency_policy\":null,\"total_memory_bytes\":null}},\"backend\":{{\"device\":\"host\",\"host_provider\":\"accelerate\",\"thread_count\":1,\"accelerator\":null}},\"setup_latency_ns\":10,\"stats\":{{\"min_ns\":10,\"median_ns\":12,\"mean_ns\":12.0,\"p95_ns\":13,\"max_ns\":13,\"throughput_per_second\":10.0,\"throughput_unit\":\"samples\"}},\"memory\":null,\"notes\":null}}\n{{\"benchmark_id\":\"model-infer.mnist.synthetic\",\"spec_path\":\"{s}\",\"suite\":\"model-train\",\"kind\":\"mnist_mlp_infer\",\"runner\":\"zig\",\"status\":\"ok\",\"dtype\":\"f32\",\"warmup_iterations\":1,\"measured_iterations\":2,\"batch_size\":8,\"seed\":7,\"shapes\":[{{\"name\":\"input\",\"dims\":[8,1,28,28]}}],\"provenance\":{{\"data_source\":\"synthetic.splitmix64\",\"preprocessing\":[\"reshape inputs to input_shape\"]}},\"runtime\":{{\"timestamp_unix_ms\":1,\"git_commit\":\"deadbeef\",\"git_dirty\":false,\"zig_version\":\"0.15.2\",\"harness_version\":\"0.1.0\"}},\"system\":{{\"os\":\"macos\",\"kernel\":\"Darwin 24.0\",\"arch\":\"aarch64\",\"cpu_model\":\"cpu\",\"cpu_logical_cores\":8,\"cpu_frequency_policy\":null,\"total_memory_bytes\":null}},\"backend\":{{\"device\":\"host\",\"host_provider\":\"accelerate\",\"thread_count\":1,\"accelerator\":null}},\"setup_latency_ns\":10,\"stats\":{{\"min_ns\":10,\"median_ns\":12,\"mean_ns\":12.0,\"p95_ns\":13,\"max_ns\":13,\"throughput_per_second\":10.0,\"throughput_unit\":\"samples\"}},\"memory\":null,\"notes\":null}}\n",
            .{ spec_rel, spec_rel },
        ),
    });

    const input_paths = try allocator.alloc([]const u8, 1);
    input_paths[0] = result_rel;

    const report = try buildReport(allocator, .{
        .input_paths = input_paths,
    });

    try std.testing.expect(report.summary.should_fail);
    try std.testing.expect(report.summary.issues >= 2);
}
