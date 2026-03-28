const std = @import("std");
const benchmarking = @import("benchmarking");

const smoke_specs = [_][]const u8{
    "benchmarks/specs/primitive/add-f32-1024x1024.json",
    "benchmarks/specs/blas/dot-f32-262144.json",
    "benchmarks/specs/autograd/dot-backward-f32-65536.json",
    "benchmarks/specs/memory/tensor-cache-cycle-f32-batch8-256x256.json",
    "benchmarks/specs/model-train/mnist-mlp-synthetic.json",
    "benchmarks/specs/model-infer/mnist-mlp-synthetic.json",
    "benchmarks/specs/model-train/char-lm-synthetic.json",
    "benchmarks/specs/model-infer/char-lm-synthetic.json",
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var specs = std.ArrayList(benchmarking.manifest.Spec){};
    for (smoke_specs) |spec_path| {
        try specs.append(allocator, try benchmarking.manifest.loadFromFile(allocator, spec_path));
    }

    const output_path = try std.fmt.allocPrint(
        allocator,
        ".zig-cache/benchmark-smoke-{d}.jsonl",
        .{std.time.milliTimestamp()},
    );
    defer std.fs.cwd().deleteFile(output_path) catch {};

    if (std.fs.path.dirname(output_path)) |dir_name| {
        try std.fs.cwd().makePath(dir_name);
    }

    const output_file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
    defer output_file.close();

    var buffer: [4096]u8 = undefined;
    var file_writer = output_file.writer(&buffer);
    const writer = &file_writer.interface;
    try benchmarking.cli.emitAll(
        allocator,
        writer,
        try specs.toOwnedSlice(allocator),
        .{},
        "0.1.0",
    );
    try writer.flush();

    const input_paths = try allocator.alloc([]const u8, 1);
    input_paths[0] = output_path;

    const report = try benchmarking.validate.buildReport(allocator, .{
        .input_paths = input_paths,
    });

    if (report.summary.records_checked != smoke_specs.len) {
        return error.BenchmarkSmokeMissingRecords;
    }
    if (report.summary.should_fail) {
        var stderr_buffer: [4096]u8 = undefined;
        var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);
        const stderr = &stderr_writer.interface;
        try benchmarking.validate.writeTextReport(stderr, report);
        try stderr.flush();
        return error.BenchmarkSmokeValidationFailed;
    }
}
