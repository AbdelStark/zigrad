const std = @import("std");
const manifest = @import("manifest.zig");
const result = @import("result.zig");
const metadata = @import("metadata.zig");
const workload = @import("workload.zig");

pub const Options = struct {
    spec_root: []const u8 = "benchmarks/specs",
    output_path: ?[]const u8 = null,
    group: []const u8 = "all",
    spec_path: ?[]const u8 = null,
    baseline: []const u8 = "none",
};

pub fn run(allocator: std.mem.Allocator, harness_version: []const u8) !void {
    const options = try parseArgs(allocator);
    const specs = try loadSpecs(allocator, options);
    if (options.output_path) |output_path| {
        if (std.fs.path.dirname(output_path)) |dir_name| {
            try std.fs.cwd().makePath(dir_name);
        }

        const file = try std.fs.cwd().createFile(output_path, .{ .truncate = true });
        defer file.close();

        var buffer: [4096]u8 = undefined;
        var file_writer = file.writer(&buffer);
        const writer = &file_writer.interface;
        try emitAll(allocator, writer, specs, options, harness_version);
        try writer.flush();
    } else {
        var buffer: [4096]u8 = undefined;
        var stdout_writer = std.fs.File.stdout().writer(&buffer);
        const writer = &stdout_writer.interface;
        try emitAll(allocator, writer, specs, options, harness_version);
        try writer.flush();
    }
}

fn emitAll(
    allocator: std.mem.Allocator,
    writer: anytype,
    specs: []const manifest.Spec,
    options: Options,
    harness_version: []const u8,
) !void {
    for (specs) |spec| {
        const run_output = try workload.run(allocator, spec);
        const snapshot = try metadata.collect(
            allocator,
            harness_version,
            spec.thread_count,
            run_output.host_blas_telemetry,
        );
        const stats = try result.SummaryStats.fromTimings(
            allocator,
            run_output.timings_ns,
            run_output.throughput_items,
            run_output.throughput_unit,
        );
        try result.writeJsonLine(writer, .{
            .benchmark_id = spec.id,
            .suite = spec.suite.asString(),
            .kind = spec.kind.asString(),
            .runner = "zig",
            .status = .ok,
            .dtype = spec.dtype.asString(),
            .warmup_iterations = spec.warmup_iterations,
            .measured_iterations = spec.measured_iterations,
            .batch_size = run_output.batch_size,
            .seed = spec.seed,
            .shapes = run_output.shapes,
            .runtime = snapshot.runtime,
            .system = snapshot.system,
            .backend = snapshot.backend,
            .setup_latency_ns = run_output.setup_latency_ns,
            .stats = stats,
            .memory = run_output.memory,
            .notes = run_output.notes,
        });

        if (std.mem.eql(u8, options.baseline, "pytorch")) {
            try runPytorchBaseline(allocator, writer, spec);
        }
    }
}

fn parseArgs(allocator: std.mem.Allocator) !Options {
    const args = try std.process.argsAlloc(allocator);
    var options = Options{};

    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];

        if (std.mem.eql(u8, arg, "--spec-root")) {
            index += 1;
            options.spec_root = args[index];
        } else if (std.mem.eql(u8, arg, "--output")) {
            index += 1;
            options.output_path = args[index];
        } else if (std.mem.eql(u8, arg, "--group")) {
            index += 1;
            options.group = args[index];
        } else if (std.mem.eql(u8, arg, "--spec")) {
            index += 1;
            options.spec_path = args[index];
        } else if (std.mem.eql(u8, arg, "--baseline")) {
            index += 1;
            options.baseline = args[index];
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
        \\Usage: benchmark [--spec-root <path>] [--output <path>] [--group primitive|blas|autograd|memory|model-train|model-infer|models|all] [--spec <path>] [--baseline none|pytorch]
        \\
    , .{});
}

fn loadSpecs(allocator: std.mem.Allocator, options: Options) ![]const manifest.Spec {
    var specs = std.ArrayList(manifest.Spec){};
    errdefer specs.deinit(allocator);

    if (options.spec_path) |spec_path| {
        try specs.append(allocator, try manifest.loadFromFile(allocator, spec_path));
    } else {
        if (matchesGroup(options.group, "primitive")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "primitive");
        }
        if (matchesGroup(options.group, "blas")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "blas");
        }
        if (matchesGroup(options.group, "autograd")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "autograd");
        }
        if (matchesGroup(options.group, "memory")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "memory");
        }
        if (matchesGroup(options.group, "model-train")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "model-train");
        }
        if (matchesGroup(options.group, "model-infer")) {
            try appendSuiteSpecs(allocator, &specs, options.spec_root, "model-infer");
        }
    }

    insertionSortSpecs(specs.items);
    return specs.toOwnedSlice(allocator);
}

fn matchesGroup(group: []const u8, suite_name: []const u8) bool {
    if (std.mem.eql(u8, group, "all")) return true;
    if (std.mem.eql(u8, group, "models")) {
        return std.mem.eql(u8, suite_name, "model-train") or std.mem.eql(u8, suite_name, "model-infer");
    }
    return std.mem.eql(u8, group, suite_name);
}

fn appendSuiteSpecs(
    allocator: std.mem.Allocator,
    specs: *std.ArrayList(manifest.Spec),
    spec_root: []const u8,
    suite_name: []const u8,
) !void {
    const suite_path = try std.fs.path.join(allocator, &.{ spec_root, suite_name });
    var dir = std.fs.cwd().openDir(suite_path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound => return,
        else => return err,
    };
    defer dir.close();

    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;

        const file_path = try std.fs.path.join(allocator, &.{ suite_path, entry.name });
        try specs.append(allocator, try manifest.loadFromFile(allocator, file_path));
    }
}

fn insertionSortSpecs(specs: []manifest.Spec) void {
    var i: usize = 1;
    while (i < specs.len) : (i += 1) {
        const current = specs[i];
        var j = i;
        while (j > 0 and std.mem.order(u8, specs[j - 1].id, current.id) == .gt) : (j -= 1) {
            specs[j] = specs[j - 1];
        }
        specs[j] = current;
    }
}

fn runPytorchBaseline(
    allocator: std.mem.Allocator,
    writer: anytype,
    spec: manifest.Spec,
) !void {
    const runner = spec.pytorch_runner orelse return;
    const child = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &.{ "python3", runner, "--spec", spec.path },
    }) catch return;
    defer allocator.free(child.stdout);
    defer allocator.free(child.stderr);

    const stdout = std.mem.trim(u8, child.stdout, " \t\r\n");
    if (stdout.len != 0) {
        try writer.writeAll(stdout);
        try writer.writeByte('\n');
    }
}
