const std = @import("std");
const build_options = @import("build_options");

pub const HostBlasProvider = enum {
    accelerate,
    openblas,
    mkl,

    pub fn name(self: HostBlasProvider) []const u8 {
        return switch (self) {
            .accelerate => "accelerate",
            .openblas => "openblas",
            .mkl => "mkl",
        };
    }

    pub fn supportsVml(self: HostBlasProvider) bool {
        return self == .mkl;
    }
};

pub const HostBackendInfo = struct {
    provider: HostBlasProvider,
    supports_vml: bool,
    supports_thread_env: bool = true,
    supports_f32: bool = true,
    supports_f64: bool = true,
};

pub fn parseName(value: []const u8) !HostBlasProvider {
    if (std.mem.eql(u8, value, "accelerate")) return .accelerate;
    if (std.mem.eql(u8, value, "openblas")) return .openblas;
    if (std.mem.eql(u8, value, "mkl")) return .mkl;
    return error.UnknownHostBlasProvider;
}

pub fn defaultForOs(os_tag: std.Target.Os.Tag) HostBlasProvider {
    return switch (os_tag) {
        .macos => .accelerate,
        .linux => .openblas,
        else => @panic("Unsupported host BLAS provider target"),
    };
}

pub fn backendInfo(provider: HostBlasProvider) HostBackendInfo {
    return .{
        .provider = provider,
        .supports_vml = provider.supportsVml(),
    };
}

pub const configured_host_blas_provider: HostBlasProvider = blk: {
    break :blk parseName(build_options.host_blas_provider) catch |err| switch (err) {
        error.UnknownHostBlasProvider => @compileError(std.fmt.comptimePrint(
            "Unsupported configured host BLAS provider '{s}'",
            .{build_options.host_blas_provider},
        )),
    };
};

pub const configured_host_backend = backendInfo(configured_host_blas_provider);

pub fn configuredProvider() HostBlasProvider {
    return configured_host_blas_provider;
}

test "parse host BLAS provider names" {
    try std.testing.expectEqual(.accelerate, try parseName("accelerate"));
    try std.testing.expectEqual(.openblas, try parseName("openblas"));
    try std.testing.expectEqual(.mkl, try parseName("mkl"));
}

test "default providers follow supported targets" {
    try std.testing.expectEqual(.accelerate, defaultForOs(.macos));
    try std.testing.expectEqual(.openblas, defaultForOs(.linux));
}

test "configured host BLAS provider is valid" {
    try std.testing.expectEqual(configured_host_blas_provider, try parseName(build_options.host_blas_provider));
}

test "configured backend info reports provider capabilities" {
    const info = configured_host_backend;
    try std.testing.expectEqual(configured_host_blas_provider, info.provider);
    try std.testing.expectEqual(configured_host_blas_provider.supportsVml(), info.supports_vml);
    try std.testing.expect(info.supports_thread_env);
    try std.testing.expect(info.supports_f32);
    try std.testing.expect(info.supports_f64);
}
