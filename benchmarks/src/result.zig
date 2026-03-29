const std = @import("std");

pub const Status = enum {
    ok,
    skipped,
    failed,
};

pub const ShapeMetadata = struct {
    name: []const u8,
    dims: []const usize,
};

pub const BenchmarkProvenance = struct {
    data_source: []const u8,
    preprocessing: []const []const u8,
};

pub const ThreadEnvironment = struct {
    veclib_maximum_threads: ?[]const u8 = null,
    openblas_num_threads: ?[]const u8 = null,
    omp_num_threads: ?[]const u8 = null,
    mkl_num_threads: ?[]const u8 = null,
    mkl_dynamic: ?[]const u8 = null,

    pub fn hasValues(self: ThreadEnvironment) bool {
        return self.veclib_maximum_threads != null or
            self.openblas_num_threads != null or
            self.omp_num_threads != null or
            self.mkl_num_threads != null or
            self.mkl_dynamic != null;
    }
};

pub const RuntimeMetadata = struct {
    timestamp_unix_ms: i64,
    git_commit: []const u8,
    git_dirty: bool,
    zig_version: []const u8,
    harness_version: []const u8,
};

pub const SystemMetadata = struct {
    os: []const u8,
    kernel: []const u8,
    arch: []const u8,
    cpu_model: []const u8,
    cpu_logical_cores: usize,
    cpu_frequency_policy: ?[]const u8 = null,
    total_memory_bytes: ?u64 = null,
};

pub const BackendMetadata = struct {
    device: []const u8,
    host_provider: []const u8,
    thread_count: ?u32 = null,
    accelerator: ?[]const u8 = null,
    cuda: ?CudaMetadata = null,
    thread_environment: ?ThreadEnvironment = null,
    host_blas_telemetry: ?HostBlasTelemetry = null,
};

pub const CudaMetadata = struct {
    device_count: u32,
    device_index: u32,
    device_name: []const u8,
    compute_capability_major: usize,
    compute_capability_minor: usize,
    multiprocessor_count: usize,
    total_global_memory_bytes: usize,
    driver_version: []const u8,
    runtime_version: []const u8,
};

pub const HostBlasTelemetry = struct {
    dot_calls: u64 = 0,
    matvec_calls: u64 = 0,
    matmul_calls: u64 = 0,
    bmm_acc_calls: u64 = 0,
    direct_bmm_dispatches: u64 = 0,
    fallback_bmm_dispatches: u64 = 0,
    fallback_bmm_batches: u64 = 0,
};

pub const SummaryStats = struct {
    min_ns: u64,
    median_ns: u64,
    mean_ns: f64,
    p95_ns: u64,
    max_ns: u64,
    throughput_per_second: ?f64 = null,
    throughput_unit: ?[]const u8 = null,

    pub fn fromTimings(
        allocator: std.mem.Allocator,
        timings_ns: []const u64,
        throughput_items: ?usize,
        throughput_unit: ?[]const u8,
    ) !SummaryStats {
        if (timings_ns.len == 0) return error.EmptyTimingSeries;

        const sorted = try allocator.dupe(u64, timings_ns);
        defer allocator.free(sorted);
        insertionSort(sorted);

        var sum: f64 = 0;
        for (timings_ns) |timing_ns| {
            sum += @as(f64, @floatFromInt(timing_ns));
        }

        const median_ns = if (sorted.len % 2 == 1)
            sorted[sorted.len / 2]
        else
            (sorted[(sorted.len / 2) - 1] + sorted[sorted.len / 2]) / 2;

        const p95_index = if (sorted.len == 1) 0 else ((sorted.len - 1) * 95) / 100;
        const mean_ns = sum / @as(f64, @floatFromInt(timings_ns.len));

        var throughput_per_second: ?f64 = null;
        if (throughput_items) |items| {
            if (mean_ns > 0) {
                const seconds = mean_ns / @as(f64, std.time.ns_per_s);
                throughput_per_second = @as(f64, @floatFromInt(items)) / seconds;
            }
        }

        return .{
            .min_ns = sorted[0],
            .median_ns = median_ns,
            .mean_ns = mean_ns,
            .p95_ns = sorted[p95_index],
            .max_ns = sorted[sorted.len - 1],
            .throughput_per_second = throughput_per_second,
            .throughput_unit = if (throughput_per_second == null) null else throughput_unit,
        };
    }
};

pub const MemoryStats = struct {
    peak_live_bytes: ?u64 = null,
    final_live_bytes: ?u64 = null,
    peak_graph_arena_bytes: ?u64 = null,
    final_graph_arena_bytes: ?u64 = null,
    peak_scratch_bytes: ?u64 = null,
};

pub const InteropMetrics = struct {
    artifact_bytes: usize,
    tensor_count: usize,
};

pub const Record = struct {
    benchmark_id: []const u8,
    spec_path: ?[]const u8 = null,
    suite: []const u8,
    kind: []const u8,
    runner: []const u8,
    status: Status,
    dtype: []const u8,
    warmup_iterations: u32,
    measured_iterations: u32,
    batch_size: ?usize,
    seed: u64,
    shapes: []const ShapeMetadata,
    provenance: ?BenchmarkProvenance = null,
    runtime: RuntimeMetadata,
    system: SystemMetadata,
    backend: BackendMetadata,
    setup_latency_ns: ?u64 = null,
    stats: ?SummaryStats = null,
    memory: ?MemoryStats = null,
    interop: ?InteropMetrics = null,
    notes: ?[]const u8 = null,
};

pub const LoadedFile = struct {
    arena: std.heap.ArenaAllocator,
    records: []const Record,

    pub fn loadFromFile(parent_allocator: std.mem.Allocator, path: []const u8) !LoadedFile {
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        const bytes = try file.readToEndAlloc(allocator, 64 * 1024 * 1024);
        return .{
            .arena = arena,
            .records = try parseJsonLines(allocator, bytes),
        };
    }

    pub fn loadFromSlice(parent_allocator: std.mem.Allocator, bytes: []const u8) !LoadedFile {
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();
        const owned_bytes = try allocator.dupe(u8, bytes);

        return .{
            .arena = arena,
            .records = try parseJsonLines(allocator, owned_bytes),
        };
    }

    pub fn deinit(self: *LoadedFile) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub fn writeJsonLine(writer: anytype, record: Record) !void {
    try std.json.Stringify.value(record, .{}, writer);
    try writer.writeByte('\n');
}

fn parseJsonLines(allocator: std.mem.Allocator, bytes: []const u8) ![]const Record {
    var records = std.ArrayList(Record){};
    errdefer records.deinit(allocator);

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;

        try records.append(
            allocator,
            try std.json.parseFromSliceLeaky(Record, allocator, line, .{
                .ignore_unknown_fields = true,
            }),
        );
    }

    return records.toOwnedSlice(allocator);
}

fn insertionSort(values: []u64) void {
    var i: usize = 1;
    while (i < values.len) : (i += 1) {
        const key = values[i];
        var j = i;
        while (j > 0 and values[j - 1] > key) : (j -= 1) {
            values[j] = values[j - 1];
        }
        values[j] = key;
    }
}

test "summary stats compute deterministic aggregates" {
    const allocator = std.testing.allocator;
    const stats = try SummaryStats.fromTimings(allocator, &.{ 5, 1, 3, 2, 4 }, 10, "samples");

    try std.testing.expectEqual(@as(u64, 1), stats.min_ns);
    try std.testing.expectEqual(@as(u64, 3), stats.median_ns);
    try std.testing.expectEqual(@as(u64, 4), stats.p95_ns);
    try std.testing.expectEqual(@as(u64, 5), stats.max_ns);
    try std.testing.expectApproxEqAbs(@as(f64, 3), stats.mean_ns, 1e-9);
    try std.testing.expect(stats.throughput_per_second != null);
}

test "load jsonl records from slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const records = try parseJsonLines(allocator,
        \\{"benchmark_id":"primitive.add.f32.1x1","spec_path":"benchmarks/specs/primitive/add.json","suite":"primitive","kind":"primitive_add","runner":"zig","status":"ok","dtype":"f32","warmup_iterations":1,"measured_iterations":2,"batch_size":null,"seed":1,"shapes":[{"name":"lhs","dims":[1]},{"name":"rhs","dims":[1]}],"provenance":{"data_source":"synthetic.splitmix64","preprocessing":["reshape lhs","reshape rhs"]},"runtime":{"timestamp_unix_ms":0,"git_commit":"deadbeef","git_dirty":false,"zig_version":"0.15.2","harness_version":"0.1.0"},"system":{"os":"linux","kernel":"test","arch":"x86_64","cpu_model":"cpu","cpu_logical_cores":1,"cpu_frequency_policy":"performance","total_memory_bytes":null},"backend":{"device":"host","host_provider":"blas","thread_count":1,"accelerator":null,"thread_environment":{"omp_num_threads":"1"}},"setup_latency_ns":10,"stats":{"min_ns":10,"median_ns":10,"mean_ns":10.0,"p95_ns":10,"max_ns":10,"throughput_per_second":100.0,"throughput_unit":"elements"},"memory":{"peak_live_bytes":64,"final_live_bytes":0,"peak_graph_arena_bytes":null,"final_graph_arena_bytes":null,"peak_scratch_bytes":32},"notes":null}
    );

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqualStrings("primitive.add.f32.1x1", records[0].benchmark_id);
    try std.testing.expectEqualStrings("benchmarks/specs/primitive/add.json", records[0].spec_path.?);
    try std.testing.expectEqual(Status.ok, records[0].status);
    try std.testing.expectEqualStrings("synthetic.splitmix64", records[0].provenance.?.data_source);
    try std.testing.expectEqualStrings("performance", records[0].system.cpu_frequency_policy.?);
    try std.testing.expectEqualStrings("1", records[0].backend.thread_environment.?.omp_num_threads.?);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), records[0].stats.?.mean_ns, 1e-9);
    try std.testing.expectEqual(@as(u64, 64), records[0].memory.?.peak_live_bytes.?);
    try std.testing.expect(records[0].interop == null);
}

test "load jsonl interop metrics from slice" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const records = try parseJsonLines(allocator,
        \\{"benchmark_id":"interop.mnist.synthetic","spec_path":"benchmarks/specs/interop/mnist.json","suite":"interop","kind":"interop_mnist_mlp_safetensors_export","runner":"zig","status":"ok","dtype":"f32","warmup_iterations":0,"measured_iterations":1,"batch_size":null,"seed":1,"shapes":[{"name":"weights.0","dims":[128,784]},{"name":"biases.0","dims":[128]}],"provenance":{"data_source":"synthetic.splitmix64","preprocessing":["materialize deterministic benchmark mnist parameters","encode affine parameter stack as safetensors bytes"]},"runtime":{"timestamp_unix_ms":1,"git_commit":"deadbeef","git_dirty":false,"zig_version":"0.15.2","harness_version":"0.1.0"},"system":{"os":"linux","kernel":"test","arch":"x86_64","cpu_model":"cpu","cpu_logical_cores":1,"cpu_frequency_policy":"performance","total_memory_bytes":null},"backend":{"device":"host","host_provider":"accelerate","thread_count":1,"accelerator":null},"setup_latency_ns":10,"stats":{"min_ns":10,"median_ns":10,"mean_ns":10.0,"p95_ns":10,"max_ns":10,"throughput_per_second":100.0,"throughput_unit":"bytes"},"memory":null,"interop":{"artifact_bytes":4096,"tensor_count":2},"notes":null}
    );

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(usize, 4096), records[0].interop.?.artifact_bytes);
    try std.testing.expectEqual(@as(usize, 2), records[0].interop.?.tensor_count);
}
