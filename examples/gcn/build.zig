const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const legacy_enable_mkl = b.option(bool, "enable_mkl", "Deprecated: use -Dhost_blas=mkl.") orelse false;
    const host_blas = b.option([]const u8, "host_blas", "Host BLAS provider: auto|accelerate|openblas|mkl.") orelse if (legacy_enable_mkl) "mkl" else "auto";

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .host_blas = host_blas,
    });

    const exe = b.addExecutable(.{
        .name = "main",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    exe.linkLibC();
    exe.root_module.addImport("zigrad", zigrad_dep.module("zigrad"));
    b.installArtifact(exe);

    const run_step = b.step("run", "run the main");
    const run_exe = b.addRunArtifact(exe);
    run_exe.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_exe.addArgs(args);
    }
    run_step.dependOn(&run_exe.step);
}
