const std = @import("std");
const benchmarking = @import("benchmarking");

const manifest = benchmarking.manifest;
const result = benchmarking.result;
const validate = benchmarking.validate;

const harness_version = "0.1.0";
const cuda_specs = [_][]const u8{
    "benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json",
    "benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json",
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const smoke_dir = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/benchmark-cuda-request-smoke-{d}",
        .{std.time.milliTimestamp()},
    );
    try std.fs.cwd().makePath(smoke_dir);
    defer std.fs.cwd().deleteTree(smoke_dir) catch {};

    const output_path = try std.fs.path.join(allocator, &.{ smoke_dir, "cuda-request-smoke.jsonl" });
    const specs = try loadSpecs(allocator, &cuda_specs);
    try emitBenchmarkResults(allocator, output_path, specs);
    try expectValidationPass(allocator, output_path, cuda_specs.len);

    var loaded = try result.LoadedFile.loadFromFile(allocator, output_path);
    defer loaded.deinit();

    if (loaded.records.len != cuda_specs.len) return error.BenchmarkCudaRequestSmokeMissingRecords;

    for (loaded.records) |record_entry| {
        try std.testing.expectEqualStrings("zig", record_entry.runner);
        try std.testing.expectEqualStrings("cuda", record_entry.backend.device);
        try std.testing.expect(record_entry.backend.accelerator != null);

        switch (record_entry.status) {
            .ok => {
                try std.testing.expect(record_entry.stats != null);
                try std.testing.expect(record_entry.setup_latency_ns != null);
                try std.testing.expect(record_entry.backend.cuda != null);
            },
            .skipped => {
                try std.testing.expect(record_entry.notes != null);
                try expectContains(record_entry.notes.?, "CUDA");
            },
            .failed => return error.BenchmarkCudaRequestSmokeUnexpectedFailure,
        }
    }
}

fn loadSpecs(
    allocator: std.mem.Allocator,
    spec_paths: []const []const u8,
) ![]manifest.Spec {
    const specs = try allocator.alloc(manifest.Spec, spec_paths.len);
    for (spec_paths, 0..) |spec_path, index| {
        specs[index] = try manifest.loadFromFile(allocator, spec_path);
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
        return error.BenchmarkCudaRequestSmokeMissingRecords;
    }
    if (report.summary.should_fail) {
        return error.BenchmarkCudaRequestSmokeValidationFailed;
    }
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        return error.ExpectedSubstringMissing;
    }
}
