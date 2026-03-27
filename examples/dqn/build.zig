const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const legacy_enable_mkl = b.option(bool, "enable_mkl", "Deprecated: use -Dhost_blas=mkl.") orelse false;
    const host_blas = b.option([]const u8, "host_blas", "Host BLAS provider: auto|accelerate|openblas|mkl.") orelse if (legacy_enable_mkl) "mkl" else "auto";

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .tracy_enable = false,
        .host_blas = host_blas,
    });
    const tensorboard_dep = b.dependency("tensorboard", .{
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "dqn",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_dep.module("zigrad") },
                .{ .name = "tensorboard", .module = tensorboard_dep.module("tensorboard") },
            },
        }),
    });
    exe.linkLibC();
    b.installArtifact(exe);
}
