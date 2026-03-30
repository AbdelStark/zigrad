const std = @import("std");
const benchmarking = @import("benchmarking");

const compare = benchmarking.compare;
const manifest = benchmarking.manifest;
const publication_bundle = benchmarking.publication_bundle;
const provider_report = benchmarking.provider_report;
const result = benchmarking.result;
const thread_report = benchmarking.thread_report;
const validate = benchmarking.validate;

const harness_version = "0.1.0";
const thread_sweep_spec_path = "benchmarks/specs/primitive/matmul-f32-256x256x256.json";
const thread_sweep_counts = [_]u32{ 1, 2 };

const RecordMutation = struct {
    host_provider: ?[]const u8 = null,
    latency_scale_numerator: u64 = 1,
    latency_scale_denominator: u64 = 1,
    note: ?[]const u8 = null,
    rewrite_thread_environment_for_openblas: bool = false,
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const smoke_dir = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/benchmark-publication-smoke-{d}",
        .{std.time.milliTimestamp()},
    );
    try std.fs.cwd().makePath(smoke_dir);
    defer std.fs.cwd().deleteTree(smoke_dir) catch {};

    const baseline_path = try std.fs.path.join(allocator, &.{ smoke_dir, "thread-sweep.jsonl" });
    const candidate_path = try std.fs.path.join(allocator, &.{ smoke_dir, "candidate.jsonl" });
    const provider_path = try std.fs.path.join(allocator, &.{ smoke_dir, "openblas.jsonl" });
    const compare_text_path = try std.fs.path.join(allocator, &.{ smoke_dir, "comparison.txt" });
    const compare_json_path = try std.fs.path.join(allocator, &.{ smoke_dir, "comparison.json" });
    const provider_markdown_path = try std.fs.path.join(allocator, &.{ smoke_dir, "provider-report.md" });
    const provider_json_path = try std.fs.path.join(allocator, &.{ smoke_dir, "provider-report.json" });
    const thread_markdown_path = try std.fs.path.join(allocator, &.{ smoke_dir, "thread-report.md" });
    const thread_json_path = try std.fs.path.join(allocator, &.{ smoke_dir, "thread-report.json" });
    const publication_manifest_path = try std.fs.path.join(allocator, &.{ smoke_dir, "publication-manifest.json" });
    const publication_summary_path = try std.fs.path.join(allocator, &.{ smoke_dir, "publication-summary.md" });

    const thread_sweep_specs = try makeThreadSweepSpecs(allocator, thread_sweep_spec_path, &thread_sweep_counts);
    try emitBenchmarkResults(allocator, baseline_path, thread_sweep_specs);
    try expectValidationPass(allocator, baseline_path, thread_sweep_counts.len);

    var baseline_loaded = try result.LoadedFile.loadFromFile(allocator, baseline_path);
    defer baseline_loaded.deinit();

    // Derive provider names from the actual build config so the test works on both macOS (accelerate) and Linux (openblas).
    const baseline_provider = if (baseline_loaded.records.len > 0) baseline_loaded.records[0].backend.host_provider else "accelerate";
    const alternate_provider: []const u8 = if (std.mem.eql(u8, baseline_provider, "openblas")) "accelerate" else "openblas";
    const rewrite_for_openblas = std.mem.eql(u8, alternate_provider, "openblas");

    try writeMutatedRecords(allocator, candidate_path, baseline_loaded.records, .{
        .latency_scale_numerator = 9,
        .latency_scale_denominator = 10,
        .note = "synthetic candidate smoke variant derived from the local thread sweep run",
    });
    try expectValidationPass(allocator, candidate_path, baseline_loaded.records.len);

    try writeMutatedRecords(allocator, provider_path, baseline_loaded.records, .{
        .host_provider = alternate_provider,
        .latency_scale_numerator = 11,
        .latency_scale_denominator = 10,
        .note = "synthetic provider smoke variant derived from the local thread sweep run",
        .rewrite_thread_environment_for_openblas = rewrite_for_openblas,
    });
    try expectValidationPass(allocator, provider_path, baseline_loaded.records.len);

    var candidate_loaded = try result.LoadedFile.loadFromFile(allocator, candidate_path);
    defer candidate_loaded.deinit();
    var provider_loaded = try result.LoadedFile.loadFromFile(allocator, provider_path);
    defer provider_loaded.deinit();

    const comparison_report = try compare.buildReport(
        allocator,
        baseline_path,
        candidate_path,
        baseline_loaded.records,
        candidate_loaded.records,
        "zig",
        .{},
    );
    try writeComparisonArtifacts(compare_text_path, compare_json_path, comparison_report);

    const parsed_comparison = try loadJsonFile(compare.Report, allocator, compare_json_path);
    if (parsed_comparison.summary.improved != baseline_loaded.records.len or parsed_comparison.summary.should_fail) {
        return error.BenchmarkPublicationSmokeComparisonSummaryMismatch;
    }

    const comparison_text = try readFileAlloc(allocator, compare_text_path);
    try expectContains(comparison_text, "Benchmark comparison");
    try expectContains(comparison_text, "primitive.matmul.f32.256x256x256");

    const provider_inputs = [_]provider_report.InputFile{
        .{ .path = baseline_path, .records = baseline_loaded.records },
        .{ .path = provider_path, .records = provider_loaded.records },
    };
    const host_provider_report = try provider_report.buildReport(
        allocator,
        &provider_inputs,
        "zig",
        baseline_provider,
    );
    try writeProviderArtifacts(allocator, provider_markdown_path, provider_json_path, host_provider_report);

    const parsed_provider_report = try loadJsonFile(provider_report.Report, allocator, provider_json_path);
    if (parsed_provider_report.summary.provider_count != 2 or
        parsed_provider_report.summary.benchmark_groups != baseline_loaded.records.len or
        parsed_provider_report.summary.comparable_entries == 0)
    {
        return error.BenchmarkPublicationSmokeProviderSummaryMismatch;
    }

    const provider_markdown = try readFileAlloc(allocator, provider_markdown_path);
    const baseline_backticked = try std.fmt.allocPrint(allocator, "`{s}`", .{baseline_provider});
    const alternate_backticked = try std.fmt.allocPrint(allocator, "`{s}`", .{alternate_provider});
    try expectContains(provider_markdown, baseline_backticked);
    try expectContains(provider_markdown, alternate_backticked);

    const thread_inputs = [_]thread_report.InputFile{
        .{ .path = baseline_path, .records = baseline_loaded.records },
    };
    const host_thread_report = try thread_report.buildReport(
        allocator,
        &thread_inputs,
        "zig",
        1,
    );
    try writeThreadArtifacts(allocator, thread_markdown_path, thread_json_path, host_thread_report);

    const parsed_thread_report = try loadJsonFile(thread_report.Report, allocator, thread_json_path);
    if (parsed_thread_report.baseline_thread_count == null or
        parsed_thread_report.baseline_thread_count.? != 1 or
        parsed_thread_report.summary.benchmark_groups != 1 or
        parsed_thread_report.summary.comparable_entries == 0)
    {
        return error.BenchmarkPublicationSmokeThreadSummaryMismatch;
    }

    const thread_markdown = try readFileAlloc(allocator, thread_markdown_path);
    try expectContains(thread_markdown, "| 1 |");
    try expectContains(thread_markdown, "| 2 |");

    const bundle = try publication_bundle.buildBundle(allocator, .{
        .candidate_result_path = candidate_path,
        .baseline_result_path = baseline_path,
        .extra_result_paths = &.{provider_path},
        .comparison_json_path = compare_json_path,
        .comparison_text_path = compare_text_path,
        .provider_report_json_path = provider_json_path,
        .provider_report_markdown_path = provider_markdown_path,
        .thread_report_json_path = thread_json_path,
        .thread_report_markdown_path = thread_markdown_path,
    });
    try writePublicationArtifacts(allocator, publication_manifest_path, publication_summary_path, bundle);

    const parsed_bundle = try loadJsonFile(publication_bundle.Bundle, allocator, publication_manifest_path);
    if (parsed_bundle.artifacts.len != 9 or
        parsed_bundle.baseline_results == null or
        parsed_bundle.extra_results.len != 1 or
        parsed_bundle.comparison == null or
        parsed_bundle.provider_report == null or
        parsed_bundle.thread_report == null)
    {
        return error.BenchmarkPublicationSmokeBundleSummaryMismatch;
    }

    const publication_summary = try readFileAlloc(allocator, publication_summary_path);
    try expectContains(publication_summary, "# Benchmark Publication Bundle");
    try expectContains(publication_summary, "Candidate results");
    try expectContains(publication_summary, "Provider report");
}

fn makeThreadSweepSpecs(
    allocator: std.mem.Allocator,
    spec_path: []const u8,
    thread_counts: []const u32,
) ![]const manifest.Spec {
    const base_spec = try manifest.loadFromFile(allocator, spec_path);
    const specs = try allocator.alloc(manifest.Spec, thread_counts.len);
    for (thread_counts, 0..) |thread_count, index| {
        var overridden = base_spec;
        overridden.thread_count = thread_count;
        specs[index] = overridden;
    }
    return specs;
}

fn emitBenchmarkResults(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    specs: []const manifest.Spec,
) !void {
    const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try benchmarking.cli.emitAll(allocator, writer, specs, .{}, harness_version);
    try writer.flush();
}

fn expectValidationPass(
    allocator: std.mem.Allocator,
    input_path: []const u8,
    expected_records: usize,
) !void {
    const input_paths = try allocator.alloc([]const u8, 1);
    input_paths[0] = input_path;

    const report = try validate.buildReport(allocator, .{
        .input_paths = input_paths,
    });

    if (report.summary.records_checked != expected_records) {
        return error.BenchmarkPublicationSmokeMissingRecords;
    }

    if (report.summary.should_fail) {
        var stderr_buffer: [4096]u8 = undefined;
        var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
        const stderr = &stderr_writer.interface;
        try validate.writeTextReport(stderr, report);
        try stderr.flush();
        return error.BenchmarkPublicationSmokeValidationFailed;
    }
}

fn writeMutatedRecords(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    records: []const result.Record,
    mutation: RecordMutation,
) !void {
    const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;

    for (records) |record_entry| {
        try result.writeJsonLine(writer, try mutateRecord(allocator, record_entry, mutation));
    }
    try writer.flush();
}

fn mutateRecord(
    allocator: std.mem.Allocator,
    record_entry: result.Record,
    mutation: RecordMutation,
) !result.Record {
    std.debug.assert(mutation.latency_scale_numerator != 0);
    std.debug.assert(mutation.latency_scale_denominator != 0);

    var mutated = record_entry;
    if (mutation.host_provider) |host_provider| {
        mutated.backend.host_provider = host_provider;
    }
    if (mutation.rewrite_thread_environment_for_openblas) {
        const thread_count = record_entry.backend.thread_count orelse 1;
        const thread_string = try std.fmt.allocPrint(allocator, "{d}", .{thread_count});
        mutated.backend.thread_environment = .{
            .openblas_num_threads = thread_string,
            .omp_num_threads = thread_string,
        };
    }
    if (mutation.note) |note| {
        mutated.notes = note;
    }
    if (record_entry.setup_latency_ns) |setup_latency_ns| {
        mutated.setup_latency_ns = scaleLatency(
            setup_latency_ns,
            mutation.latency_scale_numerator,
            mutation.latency_scale_denominator,
        );
    }
    if (record_entry.stats) |stats| {
        mutated.stats = scaleSummaryStats(
            stats,
            mutation.latency_scale_numerator,
            mutation.latency_scale_denominator,
        );
    }
    return mutated;
}

fn scaleSummaryStats(
    stats: result.SummaryStats,
    numerator: u64,
    denominator: u64,
) result.SummaryStats {
    var scaled = stats;
    scaled.min_ns = scaleLatency(stats.min_ns, numerator, denominator);
    scaled.median_ns = scaleLatency(stats.median_ns, numerator, denominator);
    scaled.mean_ns = scaleFloat(stats.mean_ns, numerator, denominator);
    scaled.p95_ns = scaleLatency(stats.p95_ns, numerator, denominator);
    scaled.max_ns = scaleLatency(stats.max_ns, numerator, denominator);
    if (stats.throughput_per_second) |throughput| {
        scaled.throughput_per_second = scaleFloat(throughput, denominator, numerator);
    }
    return scaled;
}

fn scaleLatency(value: u64, numerator: u64, denominator: u64) u64 {
    if (value == 0) return 0;
    const scaled = ((@as(u128, value) * numerator) + (denominator / 2)) / denominator;
    return @as(u64, @intCast(@max(@as(u128, 1), scaled)));
}

fn scaleFloat(value: f64, numerator: u64, denominator: u64) f64 {
    return value * @as(f64, @floatFromInt(numerator)) / @as(f64, @floatFromInt(denominator));
}

fn writeComparisonArtifacts(
    text_path: []const u8,
    json_path: []const u8,
    report: compare.Report,
) !void {
    const text_file = try std.fs.cwd().createFile(text_path, .{ .truncate = true });
    defer text_file.close();

    var text_buffer: [4096]u8 = undefined;
    var text_writer = text_file.writer(&text_buffer);
    const text = &text_writer.interface;
    try compare.writeTextReport(text, report);
    try text.flush();

    try writeJsonValue(json_path, report);
}

fn writeProviderArtifacts(
    allocator: std.mem.Allocator,
    markdown_path: []const u8,
    json_path: []const u8,
    report: provider_report.Report,
) !void {
    const markdown_file = try std.fs.cwd().createFile(markdown_path, .{ .truncate = true });
    defer markdown_file.close();

    var markdown_buffer: [4096]u8 = undefined;
    var markdown_writer = markdown_file.writer(&markdown_buffer);
    const markdown = &markdown_writer.interface;
    try provider_report.writeMarkdownReport(allocator, markdown, report);
    try markdown.flush();

    try writeJsonValue(json_path, report);
}

fn writeThreadArtifacts(
    allocator: std.mem.Allocator,
    markdown_path: []const u8,
    json_path: []const u8,
    report: thread_report.Report,
) !void {
    const markdown_file = try std.fs.cwd().createFile(markdown_path, .{ .truncate = true });
    defer markdown_file.close();

    var markdown_buffer: [4096]u8 = undefined;
    var markdown_writer = markdown_file.writer(&markdown_buffer);
    const markdown = &markdown_writer.interface;
    try thread_report.writeMarkdownReport(allocator, markdown, report);
    try markdown.flush();

    try writeJsonValue(json_path, report);
}

fn writeJsonValue(path: []const u8, value: anytype) !void {
    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var writer_impl = file.writer(&buffer);
    const writer = &writer_impl.interface;
    try std.json.Stringify.value(value, .{}, writer);
    try writer.writeByte('\n');
    try writer.flush();
}

fn writePublicationArtifacts(
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    summary_path: []const u8,
    bundle: publication_bundle.Bundle,
) !void {
    try writeJsonValue(manifest_path, bundle);

    const summary_file = try std.fs.cwd().createFile(summary_path, .{ .truncate = true });
    defer summary_file.close();

    var summary_buffer: [4096]u8 = undefined;
    var summary_writer = summary_file.writer(&summary_buffer);
    const summary = &summary_writer.interface;
    try publication_bundle.writeMarkdownSummary(allocator, summary, bundle);
    try summary.flush();
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
    return try file.readToEndAlloc(allocator, 16 * 1024 * 1024);
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        return error.BenchmarkPublicationSmokeMissingExpectedContent;
    }
}
