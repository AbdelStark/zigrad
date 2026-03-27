const std = @import("std");
const Build = std.Build;
const Module = Build.Module;
const OptimizeMode = std.builtin.OptimizeMode;

pub fn build(b: *Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    build_options.step.name = "Zigrad build options";
    const build_options_module = build_options.createModule();

    const safetensors_build_options = b.addOptions();
    safetensors_build_options.step.name = "Vendored safetensors-zg build options";
    safetensors_build_options.addOption(
        bool,
        "enable_sort",
        b.option(bool, "safetensors_enable_sort", "Sort tensors before safetensors serialization.") orelse true,
    );
    const safetensors_build_options_module = safetensors_build_options.createModule();

    build_options.addOption(
        std.log.Level,
        "log_level",
        b.option(std.log.Level, "log_level", "The Log Level to be used.") orelse .info,
    );

    const enable_mkl = b.option(bool, "enable_mkl", "Link MKL.") orelse false;
    build_options.addOption(bool, "enable_mkl", enable_mkl);

    // const enable_vml = b.option(bool, "enable_vml", "Link VML.") orelse false;
    // build_options.addOption(bool, "enable_vml", enable_vml);

    const enable_cuda = b.option(bool, "enable_cuda", "Enable CUDA backend.") orelse false;
    build_options.addOption(bool, "enable_cuda", enable_cuda);
    const rebuild_cuda: bool = b.option(bool, "rebuild_cuda", "force CUDA backend to recompile") orelse false;

    const zigrad = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
        },
    });
    const safetensors_module = b.addModule("safetensors_zg", .{
        .root_source_file = b.path("src/third_party/safetensors_zg/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = safetensors_build_options_module },
        },
    });
    zigrad.addImport("safetensors_zg", safetensors_module);

    switch (target.result.os.tag) {
        .linux => {
            // TODO: Dynamic library paths for cuda build
            // zigrad.addLibraryPath(.{ .cwd_relative = "/lib/x86_64-linux-gnu/" });
            // zigrad.addLibraryPath(.{ .cwd_relative = "/usr/lib/x86_64-linux-gnu/" });
            if (enable_mkl) zigrad.linkSystemLibrary("mkl_rt", .{}) else zigrad.linkSystemLibrary("blas", .{});
        },
        .macos => zigrad.linkFramework("Accelerate", .{}),
        else => @panic("Os not supported."),
    }

    const lib = b.addLibrary(.{
        .name = "zigrad",
        .root_module = zigrad,
    });

    lib.root_module.addImport("build_options", build_options_module);

    link(target, lib, enable_mkl);
    b.installArtifact(lib);

    if (enable_cuda) {
        const cuda = make_cuda_module(b, target, rebuild_cuda);
        zigrad.addImport("cuda", cuda);
    }

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad },
            },
        }),
    });

    link(target, exe, enable_mkl);
    b.installArtifact(exe);

    const benchmark_module = b.addModule("benchmarking", .{
        .root_source_file = b.path("benchmarks/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
            .{ .name = "build_options", .module = build_options_module },
        },
    });

    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_exe, enable_mkl);
    b.installArtifact(benchmark_exe);

    const benchmark_compare_exe = b.addExecutable(.{
        .name = "benchmark_compare",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/compare_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_compare_exe, enable_mkl);
    b.installArtifact(benchmark_compare_exe);

    const run_benchmark = b.addRunArtifact(benchmark_exe);
    run_benchmark.addArgs(&.{ "--output", "benchmarks/results/latest.jsonl" });
    if (b.args) |args| {
        run_benchmark.addArgs(args);
    }
    const benchmark_step = b.step("benchmark", "Run the benchmark harness");
    benchmark_step.dependOn(&run_benchmark.step);

    const run_benchmark_primitive = b.addRunArtifact(benchmark_exe);
    run_benchmark_primitive.addArgs(&.{ "--group", "primitive", "--output", "benchmarks/results/primitive.jsonl" });
    if (b.args) |args| {
        run_benchmark_primitive.addArgs(args);
    }
    const benchmark_primitive_step = b.step("benchmark-primitive", "Run primitive benchmark specs");
    benchmark_primitive_step.dependOn(&run_benchmark_primitive.step);

    const run_benchmark_blas = b.addRunArtifact(benchmark_exe);
    run_benchmark_blas.addArgs(&.{ "--group", "blas", "--output", "benchmarks/results/blas.jsonl" });
    if (b.args) |args| {
        run_benchmark_blas.addArgs(args);
    }
    const benchmark_blas_step = b.step("benchmark-blas", "Run BLAS benchmark specs");
    benchmark_blas_step.dependOn(&run_benchmark_blas.step);

    const run_benchmark_autograd = b.addRunArtifact(benchmark_exe);
    run_benchmark_autograd.addArgs(&.{ "--group", "autograd", "--output", "benchmarks/results/autograd.jsonl" });
    if (b.args) |args| {
        run_benchmark_autograd.addArgs(args);
    }
    const benchmark_autograd_step = b.step("benchmark-autograd", "Run autograd benchmark specs");
    benchmark_autograd_step.dependOn(&run_benchmark_autograd.step);

    const run_benchmark_memory = b.addRunArtifact(benchmark_exe);
    run_benchmark_memory.addArgs(&.{ "--group", "memory", "--output", "benchmarks/results/memory.jsonl" });
    if (b.args) |args| {
        run_benchmark_memory.addArgs(args);
    }
    const benchmark_memory_step = b.step("benchmark-memory", "Run memory benchmark specs");
    benchmark_memory_step.dependOn(&run_benchmark_memory.step);

    const run_benchmark_models = b.addRunArtifact(benchmark_exe);
    run_benchmark_models.addArgs(&.{ "--group", "models", "--output", "benchmarks/results/models.jsonl" });
    if (b.args) |args| {
        run_benchmark_models.addArgs(args);
    }
    const benchmark_models_step = b.step("benchmark-models", "Run model benchmark specs");
    benchmark_models_step.dependOn(&run_benchmark_models.step);

    const run_benchmark_compare = b.addRunArtifact(benchmark_compare_exe);
    if (b.args) |args| {
        run_benchmark_compare.addArgs(args);
    }
    const benchmark_compare_step = b.step("benchmark-compare", "Compare benchmark JSONL result files");
    benchmark_compare_step.dependOn(&run_benchmark_compare.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Arg passthru (`zig build run -- arg1 arg2 etc`)
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const unit_tests = b.addTest(.{
        .root_module = zigrad,
    });

    unit_tests.root_module.addImport("build_options", build_options_module);
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_unit_tests.step);

    const benchmark_tests = b.addTest(.{
        .name = "benchmarks",
        .root_module = benchmark_module,
    });
    const run_benchmark_tests = b.addRunArtifact(benchmark_tests);
    test_step.dependOn(&run_benchmark_tests.step);

    const safetensors_unit_tests = b.addTest(.{
        .name = "safetensors_zg",
        .root_module = safetensors_module,
    });
    const run_safetensors_unit_tests = b.addRunArtifact(safetensors_unit_tests);
    test_step.dependOn(&run_safetensors_unit_tests.step);

    const safetensors_benchmark = b.addExecutable(.{
        .name = "safetensors_benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/third_party/safetensors_zg/benchmark.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "build_options", .module = safetensors_build_options_module },
            },
        }),
    });
    const run_safetensors_benchmark = b.addRunArtifact(safetensors_benchmark);
    const safetensors_benchmark_step = b.step("safetensors-benchmark", "Run the vendored safetensors benchmark");
    safetensors_benchmark_step.dependOn(&run_safetensors_benchmark.step);

    // doc gen
    const docs_step = b.addInstallDirectory(.{
        .source_dir = lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.step.dependOn(&exe.step);

    const docs = b.step("docs", "Generate documentation");
    docs.dependOn(&docs_step.step);

    // Tracy -------------------------------------------------------------------
    const tracy_enable = b.option(bool, "tracy_enable", "Enable profiling") orelse false;
    if (tracy_enable) {
        const tracy = build_tracy(b, target).?;
        inline for (.{ zigrad, exe.root_module, unit_tests.root_module }) |e| e.addImport("tracy", tracy);
    }
}

fn link(target: Build.ResolvedTarget, exe: *Build.Step.Compile, enable_mkl: bool) void {
    switch (target.result.os.tag) {
        .linux => {
            if (enable_mkl) exe.linkSystemLibrary("mkl_rt") else exe.linkSystemLibrary("blas");
            exe.linkLibC();
        },
        .macos => exe.linkFramework("Accelerate"),
        else => @panic("Os not supported."),
    }
}

pub fn build_tracy(b: *Build, target: Build.ResolvedTarget) ?*Module {
    const optimize: OptimizeMode = b.option(OptimizeMode, "tracy_optimize_mode", "Defaults to ReleaseFast") orelse .ReleaseFast;
    const options = b.addOptions();
    const enable = true; // HACK: lazy
    options.addOption(bool, "tracy_enable", enable);
    options.addOption(bool, "tracy_allocation_enable", false);
    options.addOption(bool, "tracy_callstack_enable", true);
    options.addOption(usize, "tracy_callstack_depth", 10);

    const tracy = b.addModule("tracy", .{
        .root_source_file = b.path("src/tracy.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
        .imports = &.{
            .{ .name = "tracy_build_options", .module = options.createModule() },
        },
    });

    const tracy_src = b.lazyDependency("tracy", .{}) orelse return null;
    const tracy_c_flags = &.{ "-DTRACY_ENABLE=1", "-fno-sanitize=undefined" };
    tracy.addCSourceFile(.{ .file = tracy_src.path("public/TracyClient.cpp"), .flags = tracy_c_flags });

    const unit_tests = b.addTest(.{
        .name = "tracy_test",
        .root_module = tracy,
    });
    const test_step = b.step("tracy_test", "Run unit tests");
    const run_unit_tests = b.addRunArtifact(unit_tests);
    test_step.dependOn(&run_unit_tests.step);

    const exe = b.addExecutable(.{
        .name = "tracy_demo",
        .root_module = tracy,
    });
    const run_step = b.step("tracy_demo", "");
    const run_demo = b.addRunArtifact(exe);
    run_step.dependOn(&run_demo.step);
    return tracy;
}

pub fn make_cuda_module(b: *Build, target: Build.ResolvedTarget, rebuild_cuda: bool) *std.Build.Module {
    const here = b.path(".").getPath(b);

    const cuda = b.createModule(.{
        .root_source_file = b.path("src/cuda/root.zig"),
        .target = target,
        .link_libc = true,
    });

    const exists = amalgamate_exists(b);

    if (rebuild_cuda or !exists) {
        std.log.info("COMPILING CUDA BACKEND", .{});
        run_command(b, &.{
            "python3",
            b.pathJoin(&.{ here, "scripts", "cuda_setup.py" }),
            if (rebuild_cuda) "y" else "n",
        });
    }
    cuda.addIncludePath(b.path("src/cuda/"));
    cuda.addLibraryPath(b.path("src/cuda/build"));
    cuda.linkSystemLibrary("amalgamate", .{});
    return cuda;
}

fn amalgamate_exists(b: *Build) bool {
    const here = b.path(".").getPath(b);
    const path = b.pathJoin(&.{ here, "src", "cuda", "build", "libamalgamate.so" });
    var file = std.fs.openFileAbsolute(path, .{});
    if (file) |*_file| {
        _file.close();
        return true;
    } else |_| {
        return false;
    }
}

pub fn run_command(b: *Build, args: []const []const u8) void {
    const output = b.run(args);

    if (output.len > 0)
        std.debug.print("{s}", .{output});
}
