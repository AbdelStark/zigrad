const std = @import("std");
const compare = @import("compare.zig");
const provider_report = @import("provider_report.zig");
const result = @import("result.zig");
const thread_report = @import("thread_report.zig");

pub const ArtifactKind = enum {
    candidate_results,
    baseline_results,
    extra_results,
    comparison_json,
    comparison_text,
    provider_report_json,
    provider_report_markdown,
    thread_report_json,
    thread_report_markdown,
};

pub const Artifact = struct {
    kind: ArtifactKind,
    path: []const u8,
    format: []const u8,
    size_bytes: u64,
};

pub const StatusCounts = struct {
    ok: usize = 0,
    skipped: usize = 0,
    failed: usize = 0,
};

pub const RuntimeFingerprint = struct {
    git_commit: []const u8,
    git_dirty: bool,
    zig_version: []const u8,
    harness_version: []const u8,
};

pub const SystemFingerprint = struct {
    os: []const u8,
    kernel: []const u8,
    arch: []const u8,
    cpu_model: []const u8,
    cpu_logical_cores: usize,
};

pub const TimestampRange = struct {
    min_unix_ms: i64,
    max_unix_ms: i64,
};

pub const ResultArtifactSummary = struct {
    path: []const u8,
    size_bytes: u64,
    record_count: usize,
    unique_benchmark_ids: usize,
    statuses: StatusCounts,
    suites: []const []const u8,
    runners: []const []const u8,
    devices: []const []const u8,
    host_providers: []const []const u8,
    thread_counts: []const u32,
    null_thread_count_records: usize = 0,
    runtime: RuntimeFingerprint,
    system: SystemFingerprint,
    timestamps: TimestampRange,
};

pub const ComparisonArtifactSummary = struct {
    json_path: []const u8,
    json_size_bytes: u64,
    text_path: ?[]const u8 = null,
    text_size_bytes: ?u64 = null,
    runner_filter: ?[]const u8 = null,
    warn_threshold: f64,
    fail_threshold: f64,
    summary: compare.Summary,
};

pub const ProviderReportArtifactSummary = struct {
    json_path: []const u8,
    json_size_bytes: u64,
    markdown_path: ?[]const u8 = null,
    markdown_size_bytes: ?u64 = null,
    runner_filter: ?[]const u8 = null,
    baseline_provider: ?[]const u8 = null,
    providers: []const []const u8,
    input_paths: []const []const u8,
    summary: provider_report.Summary,
};

pub const ThreadReportArtifactSummary = struct {
    json_path: []const u8,
    json_size_bytes: u64,
    markdown_path: ?[]const u8 = null,
    markdown_size_bytes: ?u64 = null,
    runner_filter: ?[]const u8 = null,
    baseline_thread_count: ?u32 = null,
    providers: []const []const u8,
    input_paths: []const []const u8,
    summary: thread_report.Summary,
};

pub const Bundle = struct {
    generated_at_unix_ms: i64,
    artifacts: []const Artifact,
    candidate_results: ResultArtifactSummary,
    baseline_results: ?ResultArtifactSummary = null,
    extra_results: []const ResultArtifactSummary = &.{},
    comparison: ?ComparisonArtifactSummary = null,
    provider_report: ?ProviderReportArtifactSummary = null,
    thread_report: ?ThreadReportArtifactSummary = null,
};

pub const Options = struct {
    candidate_result_path: ?[]const u8 = null,
    baseline_result_path: ?[]const u8 = null,
    extra_result_paths: []const []const u8 = &.{},
    comparison_json_path: ?[]const u8 = null,
    comparison_text_path: ?[]const u8 = null,
    provider_report_json_path: ?[]const u8 = null,
    provider_report_markdown_path: ?[]const u8 = null,
    thread_report_json_path: ?[]const u8 = null,
    thread_report_markdown_path: ?[]const u8 = null,
    manifest_output_path: ?[]const u8 = null,
    summary_output_path: ?[]const u8 = null,
};

pub fn runCli(allocator: std.mem.Allocator) !void {
    const options = try parseArgs(allocator);
    const bundle = try buildBundle(allocator, options);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    try writeMarkdownSummary(allocator, stdout, bundle);
    try stdout.flush();

    if (options.summary_output_path) |path| {
        try writeMarkdownSummaryToPath(allocator, path, bundle);
    }
    if (options.manifest_output_path) |path| {
        try writeJsonManifestToPath(path, bundle);
    }
}

pub fn buildBundle(allocator: std.mem.Allocator, options: Options) !Bundle {
    const candidate_path = options.candidate_result_path orelse return error.MissingCandidateResultsPath;
    if (options.comparison_text_path != null and options.comparison_json_path == null) {
        return error.MissingComparisonJsonPath;
    }
    if (options.provider_report_markdown_path != null and options.provider_report_json_path == null) {
        return error.MissingProviderReportJsonPath;
    }
    if (options.thread_report_markdown_path != null and options.thread_report_json_path == null) {
        return error.MissingThreadReportJsonPath;
    }
    if (options.comparison_json_path != null and options.baseline_result_path == null) {
        return error.MissingBaselineResultsPath;
    }

    var artifacts = std.ArrayList(Artifact){};
    errdefer artifacts.deinit(allocator);

    var available_result_realpaths = std.StringHashMap(void).init(allocator);
    defer available_result_realpaths.deinit();

    const candidate_summary = try summarizeResultArtifact(
        allocator,
        &artifacts,
        &available_result_realpaths,
        .candidate_results,
        candidate_path,
    );

    const baseline_summary = if (options.baseline_result_path) |path|
        try summarizeResultArtifact(
            allocator,
            &artifacts,
            &available_result_realpaths,
            .baseline_results,
            path,
        )
    else
        null;

    var extra_results = std.ArrayList(ResultArtifactSummary){};
    errdefer extra_results.deinit(allocator);
    for (options.extra_result_paths) |path| {
        try extra_results.append(
            allocator,
            try summarizeResultArtifact(
                allocator,
                &artifacts,
                &available_result_realpaths,
                .extra_results,
                path,
            ),
        );
    }

    const comparison_summary = if (options.comparison_json_path) |json_path|
        try summarizeComparisonArtifact(
            allocator,
            &artifacts,
            baseline_summary.?,
            candidate_summary,
            json_path,
            options.comparison_text_path,
        )
    else
        null;

    const provider_summary = if (options.provider_report_json_path) |json_path|
        try summarizeProviderReportArtifact(
            allocator,
            &artifacts,
            &available_result_realpaths,
            json_path,
            options.provider_report_markdown_path,
        )
    else
        null;

    const thread_summary = if (options.thread_report_json_path) |json_path|
        try summarizeThreadReportArtifact(
            allocator,
            &artifacts,
            &available_result_realpaths,
            json_path,
            options.thread_report_markdown_path,
        )
    else
        null;

    return .{
        .generated_at_unix_ms = std.time.milliTimestamp(),
        .artifacts = try artifacts.toOwnedSlice(allocator),
        .candidate_results = candidate_summary,
        .baseline_results = baseline_summary,
        .extra_results = try extra_results.toOwnedSlice(allocator),
        .comparison = comparison_summary,
        .provider_report = provider_summary,
        .thread_report = thread_summary,
    };
}

pub fn writeMarkdownSummary(
    allocator: std.mem.Allocator,
    writer: anytype,
    bundle: Bundle,
) !void {
    _ = allocator;
    try writer.writeAll("# Benchmark Publication Bundle\n\n");
    try writer.print("- Generated at unix ms: `{d}`\n", .{bundle.generated_at_unix_ms});
    try writer.writeByte('\n');

    try writer.writeAll("## Result Artifacts\n");
    try writeResultSummary(writer, "Candidate results", bundle.candidate_results);
    if (bundle.baseline_results) |baseline| {
        try writeResultSummary(writer, "Baseline results", baseline);
    } else {
        try writer.writeAll("- Baseline results: not provided\n");
    }
    if (bundle.extra_results.len == 0) {
        try writer.writeAll("- Extra results: none\n");
    } else {
        for (bundle.extra_results, 0..) |extra_result, index| {
            try writer.print("- Extra results {d}: ", .{index + 1});
            try writeResultSummaryBody(writer, extra_result);
        }
    }
    try writer.writeByte('\n');

    try writer.writeAll("## Derived Reports\n");
    if (bundle.comparison) |comparison_summary| {
        try writer.print(
            "- Comparison: improved `{d}`, passing `{d}`, warned `{d}`, failed `{d}`, missing `{d}`, new `{d}`, should fail `{any}`\n",
            .{
                comparison_summary.summary.improved,
                comparison_summary.summary.passing,
                comparison_summary.summary.warned,
                comparison_summary.summary.failed,
                comparison_summary.summary.missing_candidate,
                comparison_summary.summary.new_candidate,
                comparison_summary.summary.should_fail,
            },
        );
    } else {
        try writer.writeAll("- Comparison: not provided\n");
    }
    if (bundle.provider_report) |provider_summary| {
        try writer.print(
            "- Provider report: groups `{d}`, providers `{d}`, comparable entries `{d}`, missing baseline groups `{d}`\n",
            .{
                provider_summary.summary.benchmark_groups,
                provider_summary.summary.provider_count,
                provider_summary.summary.comparable_entries,
                provider_summary.summary.baseline_missing_groups,
            },
        );
    } else {
        try writer.writeAll("- Provider report: not provided\n");
    }
    if (bundle.thread_report) |thread_summary| {
        try writer.print(
            "- Thread report: groups `{d}`, providers `{d}`, comparable entries `{d}`, missing baseline groups `{d}`\n",
            .{
                thread_summary.summary.benchmark_groups,
                thread_summary.summary.provider_count,
                thread_summary.summary.comparable_entries,
                thread_summary.summary.baseline_missing_groups,
            },
        );
    } else {
        try writer.writeAll("- Thread report: not provided\n");
    }
    try writer.writeByte('\n');

    try writer.writeAll("## Artifact Inventory\n");
    try writer.writeAll("| Kind | Format | Size bytes | Path |\n");
    try writer.writeAll("| --- | --- | ---: | --- |\n");
    for (bundle.artifacts) |artifact| {
        try writer.print(
            "| `{s}` | `{s}` | {d} | `{s}` |\n",
            .{
                artifactKindLabel(artifact.kind),
                artifact.format,
                artifact.size_bytes,
                artifact.path,
            },
        );
    }
}

fn writeResultSummary(
    writer: anytype,
    label: []const u8,
    summary: ResultArtifactSummary,
) !void {
    try writer.print("- {s}: ", .{label});
    try writeResultSummaryBody(writer, summary);
}

fn writeResultSummaryBody(
    writer: anytype,
    summary: ResultArtifactSummary,
) !void {
    try writer.print(
        "`{s}` with `{d}` records across `{d}` benchmark ids; statuses ok/skipped/failed = `{d}/{d}/{d}`; providers = ",
        .{
            summary.path,
            summary.record_count,
            summary.unique_benchmark_ids,
            summary.statuses.ok,
            summary.statuses.skipped,
            summary.statuses.failed,
        },
    );
    try writeStringList(writer, summary.host_providers);
    try writer.writeAll("; threads = ");
    try writeThreadCountList(writer, summary.thread_counts, summary.null_thread_count_records);
    try writer.print("; commit `{s}`\n", .{summary.runtime.git_commit});
}

fn writeStringList(writer: anytype, values: []const []const u8) !void {
    if (values.len == 0) {
        try writer.writeAll("none");
        return;
    }

    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.writeAll(value);
    }
}

fn writeThreadCountList(writer: anytype, values: []const u32, null_count: usize) !void {
    if (values.len == 0 and null_count == 0) {
        try writer.writeAll("none");
        return;
    }

    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(", ");
        try writer.print("{d}", .{value});
    }
    if (null_count != 0) {
        if (values.len != 0) try writer.writeAll(", ");
        try writer.print("none x{d}", .{null_count});
    }
}

fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    var extra_result_paths = std.ArrayList([]const u8){};
    errdefer extra_result_paths.deinit(allocator);

    var options = Options{};

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];

        if (std.mem.eql(u8, arg, "--candidate-jsonl")) {
            options.candidate_result_path = try nextArg(args, &index, "--candidate-jsonl");
        } else if (std.mem.eql(u8, arg, "--baseline-jsonl")) {
            options.baseline_result_path = try nextArg(args, &index, "--baseline-jsonl");
        } else if (std.mem.eql(u8, arg, "--extra-results-jsonl")) {
            try extra_result_paths.append(allocator, try nextArg(args, &index, "--extra-results-jsonl"));
        } else if (std.mem.eql(u8, arg, "--comparison-json")) {
            options.comparison_json_path = try nextArg(args, &index, "--comparison-json");
        } else if (std.mem.eql(u8, arg, "--comparison-text")) {
            options.comparison_text_path = try nextArg(args, &index, "--comparison-text");
        } else if (std.mem.eql(u8, arg, "--provider-report-json")) {
            options.provider_report_json_path = try nextArg(args, &index, "--provider-report-json");
        } else if (std.mem.eql(u8, arg, "--provider-report-markdown")) {
            options.provider_report_markdown_path = try nextArg(args, &index, "--provider-report-markdown");
        } else if (std.mem.eql(u8, arg, "--thread-report-json")) {
            options.thread_report_json_path = try nextArg(args, &index, "--thread-report-json");
        } else if (std.mem.eql(u8, arg, "--thread-report-markdown")) {
            options.thread_report_markdown_path = try nextArg(args, &index, "--thread-report-markdown");
        } else if (std.mem.eql(u8, arg, "--manifest-output")) {
            options.manifest_output_path = try nextArg(args, &index, "--manifest-output");
        } else if (std.mem.eql(u8, arg, "--summary-output")) {
            options.summary_output_path = try nextArg(args, &index, "--summary-output");
        } else if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return error.HelpPrinted;
        } else {
            return error.UnknownArgument;
        }
    }

    options.extra_result_paths = try extra_result_paths.toOwnedSlice(allocator);
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: benchmark-publication-bundle --candidate-jsonl <path> [--baseline-jsonl <path>] [--extra-results-jsonl <path> ...] [--comparison-json <path>] [--comparison-text <path>] [--provider-report-json <path>] [--provider-report-markdown <path>] [--thread-report-json <path>] [--thread-report-markdown <path>] [--manifest-output <path>] [--summary-output <path>]
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

fn summarizeResultArtifact(
    allocator: std.mem.Allocator,
    artifacts: *std.ArrayList(Artifact),
    available_result_realpaths: *std.StringHashMap(void),
    artifact_kind: ArtifactKind,
    path: []const u8,
) !ResultArtifactSummary {
    const size_bytes = try statNonEmptyFile(path);
    try appendArtifact(allocator, artifacts, artifact_kind, path, "jsonl", size_bytes);

    const realpath = try std.fs.cwd().realpathAlloc(allocator, path);
    try available_result_realpaths.put(realpath, {});

    var loaded = try result.LoadedFile.loadFromFile(allocator, path);
    defer loaded.deinit();
    return try summarizeResultRecords(allocator, path, size_bytes, loaded.records);
}

fn summarizeResultRecords(
    allocator: std.mem.Allocator,
    path: []const u8,
    size_bytes: u64,
    records: []const result.Record,
) !ResultArtifactSummary {
    if (records.len == 0) return error.EmptyResultArtifact;

    const first = records[0];
    const runtime = first.runtime;
    const system = first.system;

    var benchmark_ids = std.StringHashMap(void).init(allocator);
    defer benchmark_ids.deinit();
    var suites = std.StringHashMap(void).init(allocator);
    defer suites.deinit();
    var runners = std.StringHashMap(void).init(allocator);
    defer runners.deinit();
    var devices = std.StringHashMap(void).init(allocator);
    defer devices.deinit();
    var providers = std.StringHashMap(void).init(allocator);
    defer providers.deinit();
    var thread_counts = std.AutoHashMap(u32, void).init(allocator);
    defer thread_counts.deinit();

    var statuses = StatusCounts{};
    var null_thread_count_records: usize = 0;
    var min_unix_ms = first.runtime.timestamp_unix_ms;
    var max_unix_ms = first.runtime.timestamp_unix_ms;

    for (records) |record_entry| {
        if (!runtimeMatches(runtime, record_entry.runtime)) return error.InconsistentRuntimeMetadata;
        if (!systemMatches(system, record_entry.system)) return error.InconsistentSystemMetadata;

        try benchmark_ids.put(record_entry.benchmark_id, {});
        try suites.put(record_entry.suite, {});
        try runners.put(record_entry.runner, {});
        try devices.put(record_entry.backend.device, {});
        try providers.put(record_entry.backend.host_provider, {});

        if (record_entry.backend.thread_count) |thread_count| {
            try thread_counts.put(thread_count, {});
        } else {
            null_thread_count_records += 1;
        }

        switch (record_entry.status) {
            .ok => statuses.ok += 1,
            .skipped => statuses.skipped += 1,
            .failed => statuses.failed += 1,
        }

        min_unix_ms = @min(min_unix_ms, record_entry.runtime.timestamp_unix_ms);
        max_unix_ms = @max(max_unix_ms, record_entry.runtime.timestamp_unix_ms);
    }

    return .{
        .path = try allocator.dupe(u8, path),
        .size_bytes = size_bytes,
        .record_count = records.len,
        .unique_benchmark_ids = benchmark_ids.count(),
        .statuses = statuses,
        .suites = try collectSortedStrings(allocator, &suites),
        .runners = try collectSortedStrings(allocator, &runners),
        .devices = try collectSortedStrings(allocator, &devices),
        .host_providers = try collectSortedStrings(allocator, &providers),
        .thread_counts = try collectSortedThreadCounts(allocator, &thread_counts),
        .null_thread_count_records = null_thread_count_records,
        .runtime = .{
            .git_commit = try allocator.dupe(u8, runtime.git_commit),
            .git_dirty = runtime.git_dirty,
            .zig_version = try allocator.dupe(u8, runtime.zig_version),
            .harness_version = try allocator.dupe(u8, runtime.harness_version),
        },
        .system = .{
            .os = try allocator.dupe(u8, system.os),
            .kernel = try allocator.dupe(u8, system.kernel),
            .arch = try allocator.dupe(u8, system.arch),
            .cpu_model = try allocator.dupe(u8, system.cpu_model),
            .cpu_logical_cores = system.cpu_logical_cores,
        },
        .timestamps = .{
            .min_unix_ms = min_unix_ms,
            .max_unix_ms = max_unix_ms,
        },
    };
}

fn summarizeComparisonArtifact(
    allocator: std.mem.Allocator,
    artifacts: *std.ArrayList(Artifact),
    baseline_summary: ResultArtifactSummary,
    candidate_summary: ResultArtifactSummary,
    json_path: []const u8,
    text_path: ?[]const u8,
) !ComparisonArtifactSummary {
    const json_size_bytes = try statNonEmptyFile(json_path);
    try appendArtifact(allocator, artifacts, .comparison_json, json_path, "json", json_size_bytes);
    const comparison_report = try loadJsonFile(compare.Report, allocator, json_path);

    if (!std.mem.eql(u8, comparison_report.baseline_path, baseline_summary.path)) {
        return error.ComparisonBaselinePathMismatch;
    }
    if (!std.mem.eql(u8, comparison_report.candidate_path, candidate_summary.path)) {
        return error.ComparisonCandidatePathMismatch;
    }

    const text_size_bytes = if (text_path) |path| blk: {
        const size = try statNonEmptyFile(path);
        try appendArtifact(allocator, artifacts, .comparison_text, path, "text", size);
        break :blk size;
    } else null;

    return .{
        .json_path = try allocator.dupe(u8, json_path),
        .json_size_bytes = json_size_bytes,
        .text_path = if (text_path) |path| try allocator.dupe(u8, path) else null,
        .text_size_bytes = text_size_bytes,
        .runner_filter = try duplicateOptionalString(allocator, comparison_report.runner_filter),
        .warn_threshold = comparison_report.thresholds.warn_ratio,
        .fail_threshold = comparison_report.thresholds.fail_ratio,
        .summary = comparison_report.summary,
    };
}

fn summarizeProviderReportArtifact(
    allocator: std.mem.Allocator,
    artifacts: *std.ArrayList(Artifact),
    available_result_realpaths: *std.StringHashMap(void),
    json_path: []const u8,
    markdown_path: ?[]const u8,
) !ProviderReportArtifactSummary {
    const json_size_bytes = try statNonEmptyFile(json_path);
    try appendArtifact(allocator, artifacts, .provider_report_json, json_path, "json", json_size_bytes);
    const provider_summary = try loadJsonFile(provider_report.Report, allocator, json_path);

    try validateReportInputPaths(allocator, available_result_realpaths, provider_summary.input_paths);

    const markdown_size_bytes = if (markdown_path) |path| blk: {
        const size = try statNonEmptyFile(path);
        try appendArtifact(allocator, artifacts, .provider_report_markdown, path, "markdown", size);
        break :blk size;
    } else null;

    return .{
        .json_path = try allocator.dupe(u8, json_path),
        .json_size_bytes = json_size_bytes,
        .markdown_path = if (markdown_path) |path| try allocator.dupe(u8, path) else null,
        .markdown_size_bytes = markdown_size_bytes,
        .runner_filter = try duplicateOptionalString(allocator, provider_summary.runner_filter),
        .baseline_provider = try duplicateOptionalString(allocator, provider_summary.baseline_provider),
        .providers = try duplicateStringSlice(allocator, provider_summary.providers),
        .input_paths = try duplicateStringSlice(allocator, provider_summary.input_paths),
        .summary = provider_summary.summary,
    };
}

fn summarizeThreadReportArtifact(
    allocator: std.mem.Allocator,
    artifacts: *std.ArrayList(Artifact),
    available_result_realpaths: *std.StringHashMap(void),
    json_path: []const u8,
    markdown_path: ?[]const u8,
) !ThreadReportArtifactSummary {
    const json_size_bytes = try statNonEmptyFile(json_path);
    try appendArtifact(allocator, artifacts, .thread_report_json, json_path, "json", json_size_bytes);
    const thread_summary = try loadJsonFile(thread_report.Report, allocator, json_path);

    try validateReportInputPaths(allocator, available_result_realpaths, thread_summary.input_paths);

    const markdown_size_bytes = if (markdown_path) |path| blk: {
        const size = try statNonEmptyFile(path);
        try appendArtifact(allocator, artifacts, .thread_report_markdown, path, "markdown", size);
        break :blk size;
    } else null;

    return .{
        .json_path = try allocator.dupe(u8, json_path),
        .json_size_bytes = json_size_bytes,
        .markdown_path = if (markdown_path) |path| try allocator.dupe(u8, path) else null,
        .markdown_size_bytes = markdown_size_bytes,
        .runner_filter = try duplicateOptionalString(allocator, thread_summary.runner_filter),
        .baseline_thread_count = thread_summary.baseline_thread_count,
        .providers = try duplicateStringSlice(allocator, thread_summary.providers),
        .input_paths = try duplicateStringSlice(allocator, thread_summary.input_paths),
        .summary = thread_summary.summary,
    };
}

fn validateReportInputPaths(
    allocator: std.mem.Allocator,
    available_result_realpaths: *std.StringHashMap(void),
    input_paths: []const []const u8,
) !void {
    for (input_paths) |path| {
        const realpath = try std.fs.cwd().realpathAlloc(allocator, path);
        if (!available_result_realpaths.contains(realpath)) {
            return error.UnknownReportInputPath;
        }
    }
}

fn loadJsonFile(comptime T: type, allocator: std.mem.Allocator, path: []const u8) !T {
    const bytes = try readFileAlloc(allocator, path);
    return try std.json.parseFromSliceLeaky(T, allocator, bytes, .{
        .ignore_unknown_fields = true,
    });
}

fn readFileAlloc(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    return try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
}

fn statNonEmptyFile(path: []const u8) !u64 {
    const stat = try std.fs.cwd().statFile(path);
    if (stat.size == 0) return error.EmptyArtifactFile;
    return stat.size;
}

fn appendArtifact(
    allocator: std.mem.Allocator,
    artifacts: *std.ArrayList(Artifact),
    kind: ArtifactKind,
    path: []const u8,
    format: []const u8,
    size_bytes: u64,
) !void {
    try artifacts.append(allocator, .{
        .kind = kind,
        .path = try allocator.dupe(u8, path),
        .format = try allocator.dupe(u8, format),
        .size_bytes = size_bytes,
    });
}

fn duplicateOptionalString(allocator: std.mem.Allocator, value: ?[]const u8) !?[]const u8 {
    return if (value) |slice| try allocator.dupe(u8, slice) else null;
}

fn duplicateStringSlice(allocator: std.mem.Allocator, values: []const []const u8) ![]const []const u8 {
    const duplicated = try allocator.alloc([]const u8, values.len);
    for (values, 0..) |value, index| {
        duplicated[index] = try allocator.dupe(u8, value);
    }
    return duplicated;
}

fn runtimeMatches(lhs: result.RuntimeMetadata, rhs: result.RuntimeMetadata) bool {
    return lhs.git_dirty == rhs.git_dirty and
        std.mem.eql(u8, lhs.git_commit, rhs.git_commit) and
        std.mem.eql(u8, lhs.zig_version, rhs.zig_version) and
        std.mem.eql(u8, lhs.harness_version, rhs.harness_version);
}

fn systemMatches(lhs: result.SystemMetadata, rhs: result.SystemMetadata) bool {
    return lhs.cpu_logical_cores == rhs.cpu_logical_cores and
        std.mem.eql(u8, lhs.os, rhs.os) and
        std.mem.eql(u8, lhs.kernel, rhs.kernel) and
        std.mem.eql(u8, lhs.arch, rhs.arch) and
        std.mem.eql(u8, lhs.cpu_model, rhs.cpu_model);
}

fn collectSortedStrings(
    allocator: std.mem.Allocator,
    set: *std.StringHashMap(void),
) ![]const []const u8 {
    var values = std.ArrayList([]const u8){};
    errdefer values.deinit(allocator);

    var iterator = set.iterator();
    while (iterator.next()) |entry| {
        try values.append(allocator, try allocator.dupe(u8, entry.key_ptr.*));
    }

    const slice = try values.toOwnedSlice(allocator);
    std.mem.sort([]const u8, slice, {}, lessThanString);
    return slice;
}

fn collectSortedThreadCounts(
    allocator: std.mem.Allocator,
    set: *std.AutoHashMap(u32, void),
) ![]const u32 {
    var values = std.ArrayList(u32){};
    errdefer values.deinit(allocator);

    var iterator = set.iterator();
    while (iterator.next()) |entry| {
        try values.append(allocator, entry.key_ptr.*);
    }

    const slice = try values.toOwnedSlice(allocator);
    std.mem.sort(u32, slice, {}, lessThanU32);
    return slice;
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.order(u8, lhs, rhs) == .lt;
}

fn lessThanU32(_: void, lhs: u32, rhs: u32) bool {
    return lhs < rhs;
}

fn artifactKindLabel(kind: ArtifactKind) []const u8 {
    return switch (kind) {
        .candidate_results => "candidate-results",
        .baseline_results => "baseline-results",
        .extra_results => "extra-results",
        .comparison_json => "comparison-json",
        .comparison_text => "comparison-text",
        .provider_report_json => "provider-report-json",
        .provider_report_markdown => "provider-report-markdown",
        .thread_report_json => "thread-report-json",
        .thread_report_markdown => "thread-report-markdown",
    };
}

fn writeMarkdownSummaryToPath(
    allocator: std.mem.Allocator,
    path: []const u8,
    bundle: Bundle,
) !void {
    if (std.fs.path.dirname(path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try writeMarkdownSummary(allocator, writer, bundle);
    try writer.flush();
}

fn writeJsonManifestToPath(path: []const u8, bundle: Bundle) !void {
    if (std.fs.path.dirname(path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try std.json.Stringify.value(bundle, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

fn makeTestRecord(
    allocator: std.mem.Allocator,
    benchmark_id: []const u8,
    git_commit: []const u8,
    thread_count: ?u32,
) !result.Record {
    const shape_dims = try allocator.dupe(usize, &.{1});
    const shapes = try allocator.alloc(result.ShapeMetadata, 1);
    shapes[0] = .{
        .name = try allocator.dupe(u8, "lhs"),
        .dims = shape_dims,
    };

    return .{
        .benchmark_id = try allocator.dupe(u8, benchmark_id),
        .spec_path = try allocator.dupe(u8, "benchmarks/specs/primitive/add.json"),
        .suite = try allocator.dupe(u8, "primitive"),
        .kind = try allocator.dupe(u8, "primitive_add"),
        .runner = try allocator.dupe(u8, "zig"),
        .status = .ok,
        .dtype = try allocator.dupe(u8, "f32"),
        .warmup_iterations = 1,
        .measured_iterations = 2,
        .batch_size = null,
        .seed = 1,
        .shapes = shapes,
        .provenance = .{
            .data_source = try allocator.dupe(u8, "synthetic.splitmix64"),
            .preprocessing = try allocator.alloc([]const u8, 0),
        },
        .runtime = .{
            .timestamp_unix_ms = 1234,
            .git_commit = try allocator.dupe(u8, git_commit),
            .git_dirty = false,
            .zig_version = try allocator.dupe(u8, "0.15.2"),
            .harness_version = try allocator.dupe(u8, "0.1.0"),
        },
        .system = .{
            .os = try allocator.dupe(u8, "linux"),
            .kernel = try allocator.dupe(u8, "test"),
            .arch = try allocator.dupe(u8, "x86_64"),
            .cpu_model = try allocator.dupe(u8, "cpu"),
            .cpu_logical_cores = 4,
            .cpu_frequency_policy = try allocator.dupe(u8, "performance"),
            .total_memory_bytes = 1024,
        },
        .backend = .{
            .device = try allocator.dupe(u8, "host"),
            .host_provider = try allocator.dupe(u8, "openblas"),
            .thread_count = thread_count,
            .accelerator = null,
            .thread_environment = .{
                .openblas_num_threads = if (thread_count) |count|
                    try std.fmt.allocPrint(allocator, "{d}", .{count})
                else
                    null,
            },
        },
        .setup_latency_ns = 10,
        .stats = .{
            .min_ns = 10,
            .median_ns = 10,
            .mean_ns = 10.0,
            .p95_ns = 10,
            .max_ns = 10,
            .throughput_per_second = 100.0,
            .throughput_unit = try allocator.dupe(u8, "elements"),
        },
        .memory = .{
            .peak_live_bytes = 64,
            .final_live_bytes = 0,
            .peak_graph_arena_bytes = 32,
            .final_graph_arena_bytes = 0,
            .peak_scratch_bytes = 16,
        },
        .notes = null,
    };
}

fn writeResultArtifact(path: []const u8, records: []const result.Record) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    for (records) |record_entry| {
        try result.writeJsonLine(writer, record_entry);
    }
    try writer.flush();
}

fn writeJsonValue(path: []const u8, value: anytype) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try std.json.Stringify.value(value, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

test "result artifact summary rejects mixed runtime metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const records = try allocator.alloc(result.Record, 2);
    records[0] = try makeTestRecord(allocator, "primitive.add.f32.a", "deadbeef", 1);
    records[1] = try makeTestRecord(allocator, "primitive.add.f32.b", "cafebabe", 2);

    try std.testing.expectError(
        error.InconsistentRuntimeMetadata,
        summarizeResultRecords(allocator, "candidate.jsonl", 128, records),
    );
}

test "bundle validates comparison paths against supplied result artifacts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const temp_dir = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/publication-bundle-test-{d}",
        .{std.time.milliTimestamp()},
    );
    try std.fs.cwd().makePath(temp_dir);
    defer std.fs.cwd().deleteTree(temp_dir) catch {};

    const candidate_path = try std.fs.path.join(allocator, &.{ temp_dir, "candidate.jsonl" });
    const baseline_path = try std.fs.path.join(allocator, &.{ temp_dir, "baseline.jsonl" });
    const comparison_path = try std.fs.path.join(allocator, &.{ temp_dir, "comparison.json" });

    const candidate_records = try allocator.alloc(result.Record, 1);
    candidate_records[0] = try makeTestRecord(allocator, "primitive.add.f32.a", "deadbeef", 1);
    const baseline_records = try allocator.alloc(result.Record, 1);
    baseline_records[0] = try makeTestRecord(allocator, "primitive.add.f32.a", "deadbeef", 1);

    try writeResultArtifact(candidate_path, candidate_records);
    try writeResultArtifact(baseline_path, baseline_records);

    const bad_report: compare.Report = .{
        .baseline_path = "wrong-baseline.jsonl",
        .candidate_path = candidate_path,
        .runner_filter = "zig",
        .thresholds = .{},
        .summary = .{},
        .comparisons = &.{},
    };
    try writeJsonValue(comparison_path, bad_report);

    try std.testing.expectError(
        error.ComparisonBaselinePathMismatch,
        buildBundle(allocator, .{
            .candidate_result_path = candidate_path,
            .baseline_result_path = baseline_path,
            .comparison_json_path = comparison_path,
        }),
    );
}
