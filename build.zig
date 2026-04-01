const std = @import("std");
const Build = std.Build;
const Module = Build.Module;
const OptimizeMode = std.builtin.OptimizeMode;

const HostBlasProvider = enum {
    auto,
    accelerate,
    openblas,
    mkl,
};

const HostBlasConfig = struct {
    provider: HostBlasProvider,
    openblas_library_name: []const u8,
    mkl_runtime_library_name: []const u8,
    mkl_include_dir: ?[]const u8,
    mkl_library_dir: ?[]const u8,

    fn providerName(self: HostBlasConfig) []const u8 {
        return switch (self.provider) {
            .accelerate => "accelerate",
            .openblas => "openblas",
            .mkl => "mkl",
            .auto => unreachable,
        };
    }

    fn usesMkl(self: HostBlasConfig) bool {
        return self.provider == .mkl;
    }
};

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

    const legacy_enable_mkl = b.option(bool, "enable_mkl", "Deprecated: use -Dhost_blas=mkl.") orelse false;
    const requested_host_blas = b.option([]const u8, "host_blas", "Host BLAS provider: auto|accelerate|openblas|mkl.") orelse if (legacy_enable_mkl) "mkl" else "auto";
    const host_blas = try resolveHostBlasConfig(
        target.result.os.tag,
        requested_host_blas,
        b.option([]const u8, "openblas_library_name", "Linux OpenBLAS runtime library name.") orelse "openblas",
        b.option([]const u8, "mkl_runtime_library_name", "Linux oneMKL runtime library name.") orelse "mkl_rt",
        b.option([]const u8, "mkl_include_dir", "Path to oneMKL headers when -Dhost_blas=mkl."),
        b.option([]const u8, "mkl_library_dir", "Path to oneMKL libraries when -Dhost_blas=mkl."),
    );
    build_options.addOption(bool, "enable_mkl", host_blas.usesMkl());
    build_options.addOption([]const u8, "host_blas_provider", host_blas.providerName());

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
    configureHostBlasModule(zigrad, host_blas);
    const safetensors_module = b.addModule("safetensors_zg", .{
        .root_source_file = b.path("src/third_party/safetensors_zg/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = safetensors_build_options_module },
        },
    });
    zigrad.addImport("safetensors_zg", safetensors_module);

    const lib = b.addLibrary(.{
        .name = "zigrad",
        .root_module = zigrad,
    });

    lib.root_module.addImport("build_options", build_options_module);

    link(target, lib, host_blas);
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

    link(target, exe, host_blas);
    b.installArtifact(exe);

    // CommitLLM E2E showcase executable
    const commitllm_showcase = b.addExecutable(.{
        .name = "commitllm-showcase",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/commitllm-showcase/src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad },
            },
        }),
    });
    link(target, commitllm_showcase, host_blas);
    b.installArtifact(commitllm_showcase);

    const run_commitllm_showcase = b.addRunArtifact(commitllm_showcase);
    run_commitllm_showcase.stdio = .inherit;
    const commitllm_showcase_step = b.step("commitllm-showcase", "Run the CommitLLM E2E showcase");
    commitllm_showcase_step.dependOn(&run_commitllm_showcase.step);

    // CommitLLM cross-language showcase (Rust prover → Zig verifier)
    const commitllm_crosslang = b.addExecutable(.{
        .name = "commitllm-crosslang-showcase",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/commitllm-crosslang/src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad },
            },
        }),
    });
    link(target, commitllm_crosslang, host_blas);
    b.installArtifact(commitllm_crosslang);

    const run_commitllm_crosslang = b.addRunArtifact(commitllm_crosslang);
    run_commitllm_crosslang.stdio = .inherit;
    const commitllm_crosslang_step = b.step("commitllm-crosslang-showcase", "Run the CommitLLM cross-language showcase (Rust prover, Zig verifier)");
    commitllm_crosslang_step.dependOn(&run_commitllm_crosslang.step);

    const benchmark_module = b.addModule("benchmarking", .{
        .root_source_file = b.path("benchmarks/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
            .{ .name = "build_options", .module = build_options_module },
        },
    });
    benchmark_module.addImport("examples_mnist_model", b.createModule(.{
        .root_source_file = b.path("examples/mnist/src/model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_dqn_model", b.createModule(.{
        .root_source_file = b.path("examples/dqn/src/dqn_model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_gcn_model", b.createModule(.{
        .root_source_file = b.path("examples/gcn/src/model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_char_lm_model", b.createModule(.{
        .root_source_file = b.path("examples/char-lm/src/model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_pendulum_model", b.createModule(.{
        .root_source_file = b.path("examples/pendulum/src/model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_pendulum_dataset", b.createModule(.{
        .root_source_file = b.path("examples/pendulum/src/dataset.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_corridor_model", b.createModule(.{
        .root_source_file = b.path("examples/corridor/src/model.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    }));
    benchmark_module.addImport("examples_corridor_environment", b.createModule(.{
        .root_source_file = b.path("examples/corridor/src/environment.zig"),
        .target = target,
        .optimize = optimize,
    }));

    const protobuf_module = b.addModule("protobuf", .{
        .root_source_file = b.path("tensorboard/src/third_party/protobuf/protobuf.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tensorboard_module = b.addModule("tensorboard", .{
        .root_source_file = b.path("tensorboard/src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "protobuf", .module = protobuf_module },
        },
    });

    const example_hello_world_main_module = b.createModule(.{
        .root_source_file = b.path("examples/hello-world/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    });
    const example_mnist_main_module = b.createModule(.{
        .root_source_file = b.path("examples/mnist/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    });
    const example_dqn_train_module = b.createModule(.{
        .root_source_file = b.path("examples/dqn/src/dqn_train.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
            .{ .name = "tensorboard", .module = tensorboard_module },
        },
    });
    const example_gcn_main_module = b.createModule(.{
        .root_source_file = b.path("examples/gcn/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    });
    const example_char_lm_main_module = b.createModule(.{
        .root_source_file = b.path("examples/char-lm/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    });
    const example_pendulum_main_module = b.createModule(.{
        .root_source_file = b.path("examples/pendulum/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
        },
    });
    const example_corridor_main_module = b.createModule(.{
        .root_source_file = b.path("examples/corridor/src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
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

    link(target, benchmark_exe, host_blas);
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

    link(target, benchmark_compare_exe, host_blas);
    b.installArtifact(benchmark_compare_exe);

    const benchmark_provider_report_exe = b.addExecutable(.{
        .name = "benchmark_provider_report",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/provider_report_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_provider_report_exe, host_blas);
    b.installArtifact(benchmark_provider_report_exe);

    const benchmark_thread_report_exe = b.addExecutable(.{
        .name = "benchmark_thread_report",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/thread_report_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_thread_report_exe, host_blas);
    b.installArtifact(benchmark_thread_report_exe);

    const benchmark_publication_bundle_exe = b.addExecutable(.{
        .name = "benchmark_publication_bundle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/publication_bundle_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_publication_bundle_exe, host_blas);
    b.installArtifact(benchmark_publication_bundle_exe);

    const benchmark_validate_exe = b.addExecutable(.{
        .name = "benchmark_validate",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmarks/src/validate_main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });

    link(target, benchmark_validate_exe, host_blas);
    b.installArtifact(benchmark_validate_exe);

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

    const run_benchmark_compiler = b.addRunArtifact(benchmark_exe);
    run_benchmark_compiler.addArgs(&.{ "--group", "compiler", "--output", "benchmarks/results/compiler.jsonl" });
    if (b.args) |args| {
        run_benchmark_compiler.addArgs(args);
    }
    const benchmark_compiler_step = b.step("benchmark-compiler", "Run compiler benchmark specs");
    benchmark_compiler_step.dependOn(&run_benchmark_compiler.step);

    const run_benchmark_interop = b.addRunArtifact(benchmark_exe);
    run_benchmark_interop.addArgs(&.{ "--group", "interop", "--output", "benchmarks/results/interop.jsonl" });
    if (b.args) |args| {
        run_benchmark_interop.addArgs(args);
    }
    const benchmark_interop_step = b.step("benchmark-interop", "Run interop benchmark specs");
    benchmark_interop_step.dependOn(&run_benchmark_interop.step);

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

    const run_benchmark_provider_report = b.addRunArtifact(benchmark_provider_report_exe);
    if (b.args) |args| {
        run_benchmark_provider_report.addArgs(args);
    }
    const benchmark_provider_report_step = b.step("benchmark-provider-report", "Generate a host BLAS provider benchmark report");
    benchmark_provider_report_step.dependOn(&run_benchmark_provider_report.step);

    const run_benchmark_thread_report = b.addRunArtifact(benchmark_thread_report_exe);
    if (b.args) |args| {
        run_benchmark_thread_report.addArgs(args);
    }
    const benchmark_thread_report_step = b.step("benchmark-thread-report", "Generate a host thread-scaling benchmark report");
    benchmark_thread_report_step.dependOn(&run_benchmark_thread_report.step);

    const run_benchmark_publication_bundle = b.addRunArtifact(benchmark_publication_bundle_exe);
    if (b.args) |args| {
        run_benchmark_publication_bundle.addArgs(args);
    }
    const benchmark_publication_bundle_step = b.step("benchmark-publication-bundle", "Summarize benchmark publication artifacts into a reproducible bundle manifest");
    benchmark_publication_bundle_step.dependOn(&run_benchmark_publication_bundle.step);

    const run_benchmark_validate = b.addRunArtifact(benchmark_validate_exe);
    if (b.args) |args| {
        run_benchmark_validate.addArgs(args);
    }
    const benchmark_validate_step = b.step("benchmark-validate", "Validate benchmark specs and JSONL artifacts");
    benchmark_validate_step.dependOn(&run_benchmark_validate.step);

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

    const provider_parity_module = b.addModule("provider_parity", .{
        .root_source_file = b.path("benchmarks/src/provider_parity.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "zigrad", .module = zigrad },
            .{ .name = "build_options", .module = build_options_module },
        },
    });
    const provider_parity_tests = b.addTest(.{
        .name = "provider_parity",
        .root_module = provider_parity_module,
    });
    const run_provider_parity_tests = b.addRunArtifact(provider_parity_tests);
    const provider_parity_step = b.step(
        "test-provider-parity",
        "Run host BLAS provider numerical parity tests",
    );
    provider_parity_step.dependOn(&run_provider_parity_tests.step);
    test_step.dependOn(&run_provider_parity_tests.step);

    const example_smoke_exe = b.addExecutable(.{
        .name = "example_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/src/example_smoke_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "examples_hello_world_main", .module = example_hello_world_main_module },
                .{ .name = "examples_mnist_main", .module = example_mnist_main_module },
                .{ .name = "examples_dqn_train", .module = example_dqn_train_module },
                .{ .name = "examples_gcn_main", .module = example_gcn_main_module },
                .{ .name = "examples_char_lm_main", .module = example_char_lm_main_module },
                .{ .name = "examples_pendulum_main", .module = example_pendulum_main_module },
                .{ .name = "examples_corridor_main", .module = example_corridor_main_module },
            },
        }),
    });
    link(target, example_smoke_exe, host_blas);
    const run_example_smoke = b.addRunArtifact(example_smoke_exe);
    const example_smoke_step = b.step(
        "test-example-smoke",
        "Run runtime smoke tests for the example portfolio",
    );
    example_smoke_step.dependOn(&run_example_smoke.step);
    test_step.dependOn(&run_example_smoke.step);

    const benchmark_smoke_exe = b.addExecutable(.{
        .name = "benchmark_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/src/benchmark_smoke_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });
    link(target, benchmark_smoke_exe, host_blas);
    const run_benchmark_smoke = b.addRunArtifact(benchmark_smoke_exe);
    const benchmark_smoke_step = b.step(
        "test-benchmark-smoke",
        "Run benchmark harness smoke validation across representative suites",
    );
    benchmark_smoke_step.dependOn(&run_benchmark_smoke.step);
    test_step.dependOn(&run_benchmark_smoke.step);

    const benchmark_cuda_request_smoke_exe = b.addExecutable(.{
        .name = "benchmark_cuda_request_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/src/benchmark_cuda_request_smoke_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });
    link(target, benchmark_cuda_request_smoke_exe, host_blas);
    const run_benchmark_cuda_request_smoke = b.addRunArtifact(benchmark_cuda_request_smoke_exe);
    const benchmark_cuda_request_smoke_step = b.step(
        "test-benchmark-cuda-request-smoke",
        "Run backend-aware CUDA benchmark request smoke validation",
    );
    benchmark_cuda_request_smoke_step.dependOn(&run_benchmark_cuda_request_smoke.step);
    test_step.dependOn(&run_benchmark_cuda_request_smoke.step);

    const benchmark_publication_smoke_exe = b.addExecutable(.{
        .name = "benchmark_publication_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/src/benchmark_publication_smoke_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });
    link(target, benchmark_publication_smoke_exe, host_blas);
    const run_benchmark_publication_smoke = b.addRunArtifact(benchmark_publication_smoke_exe);
    const benchmark_publication_smoke_step = b.step(
        "test-benchmark-publication-smoke",
        "Run benchmark publication artifact smoke validation",
    );
    benchmark_publication_smoke_step.dependOn(&run_benchmark_publication_smoke.step);
    test_step.dependOn(&run_benchmark_publication_smoke.step);

    const benchmark_baseline_smoke_exe = b.addExecutable(.{
        .name = "benchmark_baseline_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/src/benchmark_baseline_smoke_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "benchmarking", .module = benchmark_module },
            },
        }),
    });
    link(target, benchmark_baseline_smoke_exe, host_blas);
    const run_benchmark_baseline_smoke = b.addRunArtifact(benchmark_baseline_smoke_exe);
    const benchmark_baseline_smoke_step = b.step(
        "test-benchmark-baseline-smoke",
        "Run benchmark external baseline contract smoke validation",
    );
    benchmark_baseline_smoke_step.dependOn(&run_benchmark_baseline_smoke.step);
    test_step.dependOn(&run_benchmark_baseline_smoke.step);

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

fn configureHostBlasModule(module: *Build.Module, host_blas: HostBlasConfig) void {
    if (host_blas.provider == .mkl) {
        if (host_blas.mkl_include_dir) |dir| {
            module.addIncludePath(.{ .cwd_relative = dir });
        }
    }
}

fn link(target: Build.ResolvedTarget, exe: *Build.Step.Compile, host_blas: HostBlasConfig) void {
    switch (target.result.os.tag) {
        .linux => {
            switch (host_blas.provider) {
                .openblas => exe.linkSystemLibrary(host_blas.openblas_library_name),
                .mkl => {
                    if (host_blas.mkl_library_dir) |dir| {
                        exe.addLibraryPath(.{ .cwd_relative = dir });
                    }
                    exe.linkSystemLibrary(host_blas.mkl_runtime_library_name);
                },
                .accelerate, .auto => @panic("Unsupported host BLAS provider for Linux"),
            }
            exe.linkLibC();
        },
        .macos => {
            if (host_blas.provider != .accelerate) @panic("Unsupported host BLAS provider for macOS");
            exe.linkFramework("Accelerate");
        },
        else => @panic("Os not supported."),
    }
}

fn resolveHostBlasConfig(
    os_tag: std.Target.Os.Tag,
    requested_name: []const u8,
    openblas_library_name: []const u8,
    mkl_runtime_library_name: []const u8,
    mkl_include_dir: ?[]const u8,
    mkl_library_dir: ?[]const u8,
) !HostBlasConfig {
    const requested = parseHostBlasProvider(requested_name) catch {
        std.debug.print(
            "error: unknown host BLAS provider '{s}'; expected auto, accelerate, openblas, or mkl\n",
            .{requested_name},
        );
        return error.InvalidHostBlasProvider;
    };

    const provider = switch (requested) {
        .auto => defaultHostBlasProvider(os_tag),
        else => requested,
    };

    switch (os_tag) {
        .linux => switch (provider) {
            .openblas, .mkl => {},
            else => {
                std.debug.print(
                    "error: Linux host builds support only openblas or mkl; got '{s}'\n",
                    .{providerName(provider)},
                );
                return error.UnsupportedHostBlasProvider;
            },
        },
        .macos => if (provider != .accelerate) {
            std.debug.print(
                "error: macOS host builds support only accelerate; got '{s}'\n",
                .{providerName(provider)},
            );
            return error.UnsupportedHostBlasProvider;
        },
        else => {
            std.debug.print(
                "error: unsupported target OS '{s}' for host BLAS selection\n",
                .{@tagName(os_tag)},
            );
            return error.UnsupportedHostBlasProvider;
        },
    }

    return .{
        .provider = provider,
        .openblas_library_name = openblas_library_name,
        .mkl_runtime_library_name = mkl_runtime_library_name,
        .mkl_include_dir = mkl_include_dir,
        .mkl_library_dir = mkl_library_dir,
    };
}

fn parseHostBlasProvider(value: []const u8) !HostBlasProvider {
    if (std.mem.eql(u8, value, "auto")) return .auto;
    if (std.mem.eql(u8, value, "accelerate")) return .accelerate;
    if (std.mem.eql(u8, value, "openblas")) return .openblas;
    if (std.mem.eql(u8, value, "mkl")) return .mkl;
    return error.InvalidHostBlasProvider;
}

fn defaultHostBlasProvider(os_tag: std.Target.Os.Tag) HostBlasProvider {
    return switch (os_tag) {
        .macos => .accelerate,
        .linux => .openblas,
        else => @panic("Unsupported target OS for host BLAS selection"),
    };
}

fn providerName(provider: HostBlasProvider) []const u8 {
    return switch (provider) {
        .auto => "auto",
        .accelerate => "accelerate",
        .openblas => "openblas",
        .mkl => "mkl",
    };
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
