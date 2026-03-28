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
    thread_counts: []const u32 = &.{},
};

pub fn run(allocator: std.mem.Allocator, harness_version: []const u8) !void {
    const options = try parseArgs(allocator);
    const base_specs = try loadSpecs(allocator, options);
    const specs = try applyThreadCountOverrides(allocator, base_specs, options.thread_counts);
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

pub fn emitAll(
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
            .spec_path = spec.path,
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
            .provenance = spec.provenance,
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
    var thread_counts = std.ArrayList(u32){};
    errdefer thread_counts.deinit(allocator);
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
        } else if (std.mem.eql(u8, arg, "--thread-count")) {
            index += 1;
            const thread_count = try std.fmt.parseInt(u32, args[index], 10);
            if (thread_count == 0) return error.InvalidThreadCount;
            try thread_counts.append(allocator, thread_count);
        } else if (std.mem.eql(u8, arg, "--help")) {
            printUsage();
            return error.HelpPrinted;
        } else {
            return error.UnknownArgument;
        }
    }

    options.thread_counts = try thread_counts.toOwnedSlice(allocator);
    return options;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: benchmark [--spec-root <path>] [--output <path>] [--group primitive|blas|autograd|memory|model-train|model-infer|models|all] [--spec <path>] [--baseline none|pytorch] [--thread-count <n> ...]
        \\
    , .{});
}

pub fn loadSpecs(allocator: std.mem.Allocator, options: Options) ![]const manifest.Spec {
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

fn applyThreadCountOverrides(
    allocator: std.mem.Allocator,
    specs: []const manifest.Spec,
    thread_counts: []const u32,
) ![]const manifest.Spec {
    if (thread_counts.len == 0) return specs;

    var expanded = std.ArrayList(manifest.Spec){};
    errdefer expanded.deinit(allocator);

    try expanded.ensureTotalCapacity(allocator, @as(u32, @intCast(specs.len * thread_counts.len)));
    for (specs) |spec| {
        for (thread_counts) |thread_count| {
            var overridden = spec;
            overridden.thread_count = thread_count;
            try expanded.append(allocator, overridden);
        }
    }

    insertionSortSpecs(expanded.items);
    return expanded.toOwnedSlice(allocator);
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
        while (j > 0 and compareSpecs(specs[j - 1], current) == .gt) : (j -= 1) {
            specs[j] = specs[j - 1];
        }
        specs[j] = current;
    }
}

fn compareSpecs(lhs: manifest.Spec, rhs: manifest.Spec) std.math.Order {
    const id_order = std.mem.order(u8, lhs.id, rhs.id);
    if (id_order != .eq) return id_order;
    return compareThreadCounts(lhs.thread_count, rhs.thread_count);
}

fn compareThreadCounts(lhs: ?u32, rhs: ?u32) std.math.Order {
    if (lhs) |lhs_count| {
        if (rhs) |rhs_count| return std.math.order(lhs_count, rhs_count);
        return .gt;
    }
    if (rhs != null) return .lt;
    return .eq;
}

fn runPytorchBaseline(
    allocator: std.mem.Allocator,
    writer: anytype,
    spec: manifest.Spec,
) !void {
    const runner = spec.pytorch_runner orelse return;

    var argv = std.ArrayList([]const u8){};
    defer argv.deinit(allocator);
    try argv.appendSlice(allocator, &.{ "python3", runner, "--spec", spec.path });
    if (spec.thread_count) |thread_count| {
        try argv.append(allocator, "--thread-count");
        try argv.append(allocator, try std.fmt.allocPrint(allocator, "{d}", .{thread_count}));
    }

    const child = std.process.Child.run(.{
        .allocator = allocator,
        .argv = try argv.toOwnedSlice(allocator),
    }) catch return;
    defer allocator.free(child.stdout);
    defer allocator.free(child.stderr);

    const stdout = std.mem.trim(u8, child.stdout, " \t\r\n");
    if (stdout.len != 0) {
        try writer.writeAll(stdout);
        try writer.writeByte('\n');
    }
}

test "thread count overrides expand and sort specs by id then thread count" {
    const allocator = std.testing.allocator;

    const specs = [_]manifest.Spec{
        .{
            .id = "bench.b",
            .suite = .primitive,
            .kind = .primitive_add,
            .dtype = .f32,
            .provenance = .{
                .data_source = "synthetic.splitmix64",
                .preprocessing = &.{ "reshape lhs", "reshape rhs" },
            },
            .path = "inline",
        },
        .{
            .id = "bench.a",
            .suite = .primitive,
            .kind = .primitive_add,
            .dtype = .f32,
            .provenance = .{
                .data_source = "synthetic.splitmix64",
                .preprocessing = &.{ "reshape lhs", "reshape rhs" },
            },
            .path = "inline",
        },
    };

    const expanded = try applyThreadCountOverrides(allocator, specs[0..], &.{ 4, 1 });
    defer allocator.free(expanded);

    try std.testing.expectEqual(@as(usize, 4), expanded.len);
    try std.testing.expectEqualStrings("bench.a", expanded[0].id);
    try std.testing.expectEqual(@as(u32, 1), expanded[0].thread_count.?);
    try std.testing.expectEqualStrings("bench.a", expanded[1].id);
    try std.testing.expectEqual(@as(u32, 4), expanded[1].thread_count.?);
    try std.testing.expectEqualStrings("bench.b", expanded[2].id);
    try std.testing.expectEqual(@as(u32, 1), expanded[2].thread_count.?);
    try std.testing.expectEqualStrings("bench.b", expanded[3].id);
    try std.testing.expectEqual(@as(u32, 4), expanded[3].thread_count.?);
}
