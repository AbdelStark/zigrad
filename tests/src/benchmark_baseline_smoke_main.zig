const std = @import("std");
const benchmarking = @import("benchmarking");

const compare = benchmarking.compare;
const manifest = benchmarking.manifest;
const result = benchmarking.result;
const validate = benchmarking.validate;

const harness_version = "0.1.0";
const baseline_mode_env_name: [*:0]const u8 = "ZIGRAD_BASELINE_SMOKE_MODE";
const fixture_runner_path = "tests/fixtures/benchmark_baseline_smoke_runner.py";
const missing_runner_path = "tests/fixtures/does-not-exist.py";
const success_specs = [_][]const u8{
    "benchmarks/specs/blas/dot-f32-262144.json",
    "benchmarks/specs/model-infer/mnist-mlp-synthetic.json",
    "benchmarks/specs/model-train/corridor-control-synthetic.json",
    "benchmarks/specs/model-infer/corridor-control-synthetic.json",
    "benchmarks/specs/compiler/corridor-control-capture-synthetic.json",
};
const failure_spec = "benchmarks/specs/blas/dot-f32-262144.json";

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const smoke_dir = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/benchmark-baseline-smoke-{d}",
        .{std.time.milliTimestamp()},
    );
    try std.fs.cwd().makePath(smoke_dir);
    defer std.fs.cwd().deleteTree(smoke_dir) catch {};

    try runSuccessfulBaselineScenario(allocator, smoke_dir);
    try runInvalidJsonScenario(allocator, smoke_dir);
    try runMissingRunnerScenario(allocator, smoke_dir);
}

fn runSuccessfulBaselineScenario(
    allocator: std.mem.Allocator,
    smoke_dir: []const u8,
) !void {
    try setBaselineMode("ok");
    defer clearBaselineMode();

    const output_path = try std.fs.path.join(allocator, &.{ smoke_dir, "baseline-ok.jsonl" });
    const candidate_path = try std.fs.path.join(allocator, &.{ smoke_dir, "baseline-ok-candidate.jsonl" });

    const specs = try loadSpecsWithRunnerOverride(allocator, &success_specs, fixture_runner_path);
    try emitBenchmarkResults(allocator, output_path, specs, .{ .baseline = "pytorch" });
    try expectValidationPass(allocator, output_path, success_specs.len * 2);

    var loaded = try result.LoadedFile.loadFromFile(allocator, output_path);
    defer loaded.deinit();
    try expectRunnerPairing(loaded.records, success_specs.len);

    try writeMutatedPytorchCandidate(candidate_path, loaded.records, "synthetic faster baseline candidate");
    try expectValidationPass(allocator, candidate_path, loaded.records.len);

    var candidate_loaded = try result.LoadedFile.loadFromFile(allocator, candidate_path);
    defer candidate_loaded.deinit();

    const report = try compare.buildReport(
        allocator,
        output_path,
        candidate_path,
        loaded.records,
        candidate_loaded.records,
        "pytorch",
        .{},
    );

    if (report.summary.improved != success_specs.len or report.summary.should_fail) {
        return error.BenchmarkBaselineSmokeComparisonMismatch;
    }
}

fn runInvalidJsonScenario(
    allocator: std.mem.Allocator,
    smoke_dir: []const u8,
) !void {
    try setBaselineMode("invalid-json");
    defer clearBaselineMode();

    const output_path = try std.fs.path.join(allocator, &.{ smoke_dir, "baseline-invalid-json.jsonl" });
    const specs = try loadSpecsWithRunnerOverride(allocator, &.{failure_spec}, fixture_runner_path);
    try emitBenchmarkResults(allocator, output_path, specs, .{ .baseline = "pytorch" });
    try expectValidationPass(allocator, output_path, 2);

    var loaded = try result.LoadedFile.loadFromFile(allocator, output_path);
    defer loaded.deinit();

    const pytorch_record = findRunnerRecord(loaded.records, failure_spec, "pytorch") orelse
        return error.BenchmarkBaselineSmokeMissingFailureRecord;
    try std.testing.expectEqual(result.Status.failed, pytorch_record.status);
    try expectContains(pytorch_record.notes orelse return error.BenchmarkBaselineSmokeMissingFailureNote, "invalid JSONL");
}

fn runMissingRunnerScenario(
    allocator: std.mem.Allocator,
    smoke_dir: []const u8,
) !void {
    clearBaselineMode();

    const output_path = try std.fs.path.join(allocator, &.{ smoke_dir, "baseline-missing-runner.jsonl" });
    const specs = try loadSpecsWithRunnerOverride(allocator, &.{failure_spec}, missing_runner_path);
    try emitBenchmarkResults(allocator, output_path, specs, .{ .baseline = "pytorch" });
    try expectValidationPass(allocator, output_path, 2);

    var loaded = try result.LoadedFile.loadFromFile(allocator, output_path);
    defer loaded.deinit();

    const pytorch_record = findRunnerRecord(loaded.records, failure_spec, "pytorch") orelse
        return error.BenchmarkBaselineSmokeMissingFailureRecord;
    try std.testing.expectEqual(result.Status.failed, pytorch_record.status);
    try expectContains(pytorch_record.notes orelse return error.BenchmarkBaselineSmokeMissingFailureNote, "exited with code");
}

fn loadSpecsWithRunnerOverride(
    allocator: std.mem.Allocator,
    spec_paths: []const []const u8,
    runner_path: []const u8,
) ![]manifest.Spec {
    const specs = try allocator.alloc(manifest.Spec, spec_paths.len);
    for (spec_paths, 0..) |spec_path, index| {
        var spec = try manifest.loadFromFile(allocator, spec_path);
        spec.pytorch_runner = runner_path;
        specs[index] = spec;
    }
    return specs;
}

fn emitBenchmarkResults(
    allocator: std.mem.Allocator,
    output_path: []const u8,
    specs: []const manifest.Spec,
    options: benchmarking.cli.Options,
) !void {
    const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;
    try benchmarking.cli.emitAll(allocator, writer, specs, options, harness_version);
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
        return error.BenchmarkBaselineSmokeMissingRecords;
    }
    if (report.summary.should_fail) {
        return error.BenchmarkBaselineSmokeValidationFailed;
    }
}

fn expectRunnerPairing(records: []const result.Record, expected_benchmarks: usize) !void {
    var zig_count: usize = 0;
    var pytorch_count: usize = 0;
    for (records) |record_entry| {
        if (std.mem.eql(u8, record_entry.runner, "zig")) {
            zig_count += 1;
            continue;
        }
        if (std.mem.eql(u8, record_entry.runner, "pytorch")) {
            pytorch_count += 1;
            try std.testing.expectEqual(result.Status.ok, record_entry.status);
            try std.testing.expectEqualStrings("cpu", record_entry.backend.device);
            continue;
        }
        return error.BenchmarkBaselineSmokeUnexpectedRunner;
    }

    try std.testing.expectEqual(expected_benchmarks, zig_count);
    try std.testing.expectEqual(expected_benchmarks, pytorch_count);
}

fn writeMutatedPytorchCandidate(
    output_path: []const u8,
    records: []const result.Record,
    note: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = file.writer(&buffer);
    const writer = &file_writer.interface;

    for (records) |record_entry| {
        var mutated = record_entry;
        if (std.mem.eql(u8, record_entry.runner, "pytorch")) {
            mutated.notes = note;
            if (record_entry.setup_latency_ns) |setup_latency_ns| {
                mutated.setup_latency_ns = scaleLatency(setup_latency_ns, 9, 10);
            }
            if (record_entry.stats) |stats| {
                mutated.stats = scaleSummaryStats(stats, 9, 10);
            }
        }
        try result.writeJsonLine(writer, mutated);
    }
    try writer.flush();
}

fn scaleSummaryStats(
    stats: result.SummaryStats,
    numerator: u64,
    denominator: u64,
) result.SummaryStats {
    var scaled = stats;
    scaled.min_ns = scaleLatency(stats.min_ns, numerator, denominator);
    scaled.median_ns = scaleLatency(stats.median_ns, numerator, denominator);
    scaled.mean_ns = (@as(f64, @floatFromInt(numerator)) * stats.mean_ns) /
        @as(f64, @floatFromInt(denominator));
    scaled.p95_ns = scaleLatency(stats.p95_ns, numerator, denominator);
    scaled.max_ns = scaleLatency(stats.max_ns, numerator, denominator);
    if (stats.throughput_per_second) |throughput| {
        scaled.throughput_per_second =
            throughput * @as(f64, @floatFromInt(denominator)) /
            @as(f64, @floatFromInt(numerator));
    }
    return scaled;
}

fn scaleLatency(
    value: u64,
    numerator: u64,
    denominator: u64,
) u64 {
    return @max(1, @divFloor(value * numerator, denominator));
}

fn findRunnerRecord(
    records: []const result.Record,
    spec_path: []const u8,
    runner: []const u8,
) ?result.Record {
    const benchmark_id = benchmarkIdForSpec(spec_path);
    for (records) |record_entry| {
        if (std.mem.eql(u8, record_entry.benchmark_id, benchmark_id) and
            std.mem.eql(u8, record_entry.runner, runner))
        {
            return record_entry;
        }
    }
    return null;
}

fn benchmarkIdForSpec(spec_path: []const u8) []const u8 {
    if (std.mem.eql(u8, spec_path, "benchmarks/specs/blas/dot-f32-262144.json")) {
        return "blas.dot.f32.262144";
    }
    if (std.mem.eql(u8, spec_path, "benchmarks/specs/model-infer/mnist-mlp-synthetic.json")) {
        return "model-infer.mnist-mlp.synthetic.f32.batch64";
    }
    if (std.mem.eql(u8, spec_path, "benchmarks/specs/model-train/corridor-control-synthetic.json")) {
        return "model-train.corridor-control.synthetic.f32.batch24";
    }
    if (std.mem.eql(u8, spec_path, "benchmarks/specs/model-infer/corridor-control-synthetic.json")) {
        return "model-infer.corridor-control.synthetic.f32.batch24";
    }
    if (std.mem.eql(u8, spec_path, "benchmarks/specs/compiler/corridor-control-capture-synthetic.json")) {
        return "compiler.corridor-control.capture.synthetic.f32.batch24";
    }
    unreachable;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        return error.BenchmarkBaselineSmokeExpectedSubstringMissing;
    }
}

fn setBaselineMode(mode: []const u8) !void {
    var value: [32:0]u8 = undefined;
    const value_z = try std.fmt.bufPrintZ(&value, "{s}", .{mode});
    _ = setenv(baseline_mode_env_name, value_z, 1);
}

fn clearBaselineMode() void {
    _ = unsetenv(baseline_mode_env_name);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;
