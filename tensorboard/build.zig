const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tensorboard = b.addModule("tensorboard", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const protobuf = b.addModule("protobuf", .{
        .root_source_file = b.path("src/third_party/protobuf/protobuf.zig"),
        .target = target,
        .optimize = optimize,
    });
    tensorboard.addImport("protobuf", protobuf);

    const lib = b.addLibrary(.{
        .name = "tensorboard",
        .root_module = tensorboard,
        .linkage = .static,
    });

    b.installArtifact(lib);

    const lib_unit_tests = b.addTest(.{
        .root_module = tensorboard,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
