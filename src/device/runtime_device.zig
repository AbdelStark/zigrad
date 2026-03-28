const std = @import("std");
const build_options = @import("build_options");
const HostDevice = @import("host_device.zig");
const DeviceReference = @import("device_reference.zig");

const CudaDevice = if (build_options.enable_cuda)
    @import("cuda_device.zig")
else
    struct {
        pub const CudaDiagnosticsFormatOptions = struct {
            label: ?[]const u8 = null,
            trailing_newline: bool = true,
        };
    };

const log = std.log.scoped(.zg_runtime_device);

pub const RuntimeDeviceKind = enum {
    host,
    cuda,
};

pub const RuntimeDeviceRequest = struct {
    kind: RuntimeDeviceKind = .host,
    cuda_device_index: u32 = 0,
};

pub const RuntimeDeviceSupport = struct {
    allow_cuda: bool = true,
};

pub const RuntimeDeviceDiagnosticsOptions = struct {
    label: ?[]const u8 = null,
    include_telemetry: bool = true,
    trailing_newline: bool = true,
};

pub const RuntimeDevice = struct {
    kind: RuntimeDeviceKind,
    host: ?HostDevice = null,
    cuda: if (build_options.enable_cuda) ?CudaDevice else void = if (build_options.enable_cuda) null else {},

    pub fn reference(self: *RuntimeDevice) DeviceReference {
        return switch (self.kind) {
            .host => if (self.host) |*host| host.reference() else unreachable,
            .cuda => if (build_options.enable_cuda) blk: {
                if (self.cuda) |*cuda_device| break :blk cuda_device.reference();
                unreachable;
            } else unreachable,
        };
    }

    pub fn deinit(self: *RuntimeDevice) void {
        switch (self.kind) {
            .host => if (self.host) |*host| host.deinit(),
            .cuda => if (build_options.enable_cuda) {
                if (self.cuda) |*cuda_device| cuda_device.deinit();
            },
        }
        self.* = undefined;
    }

    pub fn isHost(self: *const RuntimeDevice) bool {
        return self.kind == .host;
    }

    pub fn maybeWriteRuntimeDiagnostics(
        self: *const RuntimeDevice,
        writer: anytype,
        options: RuntimeDeviceDiagnosticsOptions,
    ) !bool {
        return switch (self.kind) {
            .host => if (self.host) |*host| host.maybeWriteRuntimeDiagnostics(writer, .{
                .label = options.label,
                .include_telemetry = options.include_telemetry,
                .trailing_newline = options.trailing_newline,
            }) else false,
            .cuda => if (build_options.enable_cuda) blk: {
                if (self.cuda) |*cuda_device| {
                    break :blk try cuda_device.maybeWriteRuntimeDiagnostics(writer, .{
                        .label = options.label,
                        .trailing_newline = options.trailing_newline,
                    });
                }
                break :blk false;
            } else false,
        };
    }
};

pub fn parseRuntimeDeviceRequest(raw: ?[]const u8) !RuntimeDeviceRequest {
    const trimmed = std.mem.trim(u8, raw orelse return .{}, " \t\r\n");
    if (trimmed.len == 0) return .{};

    if (std.ascii.eqlIgnoreCase(trimmed, "host") or std.ascii.eqlIgnoreCase(trimmed, "cpu")) {
        return .{ .kind = .host };
    }

    if (std.ascii.eqlIgnoreCase(trimmed, "cuda")) {
        return .{ .kind = .cuda };
    }

    if (trimmed.len > 5 and std.ascii.eqlIgnoreCase(trimmed[0..5], "cuda:")) {
        const index = std.fmt.parseInt(u32, trimmed[5..], 10) catch return error.InvalidRuntimeDeviceRequest;
        return .{
            .kind = .cuda,
            .cuda_device_index = index,
        };
    }

    return error.InvalidRuntimeDeviceRequest;
}

pub fn resolveRuntimeDeviceRequest(explicit: ?RuntimeDeviceRequest) !RuntimeDeviceRequest {
    return explicit orelse try parseRuntimeDeviceRequest(std.posix.getenv("ZG_DEVICE"));
}

pub fn initRuntimeDevice(
    explicit: ?RuntimeDeviceRequest,
    support: RuntimeDeviceSupport,
) !RuntimeDevice {
    const request = try resolveRuntimeDeviceRequest(explicit);
    switch (request.kind) {
        .host => {
            return .{
                .kind = .host,
                .host = HostDevice.init(),
            };
        },
        .cuda => {
            if (!support.allow_cuda) {
                log.err("CUDA was requested, but this entrypoint is currently host-only.", .{});
                return error.CudaUnsupported;
            }
            if (!build_options.enable_cuda) {
                log.err("CUDA was requested, but this build was not compiled with -Denable_cuda=true.", .{});
                return error.CudaNotEnabled;
            }

            const available = CudaDevice.device_count();
            if (available == 0) {
                log.err("CUDA was requested, but no CUDA devices were detected.", .{});
                return error.CudaUnavailable;
            }
            if (request.cuda_device_index >= available) {
                log.err(
                    "CUDA device {d} was requested, but only {d} device(s) are available.",
                    .{ request.cuda_device_index, available },
                );
                return error.InvalidCudaDeviceIndex;
            }

            return .{
                .kind = .cuda,
                .host = null,
                .cuda = if (build_options.enable_cuda) CudaDevice.init(request.cuda_device_index) else unreachable,
            };
        },
    }
}

test "runtime device parsing accepts host aliases and cuda indices" {
    try std.testing.expectEqualDeep(RuntimeDeviceRequest{}, try parseRuntimeDeviceRequest(null));
    try std.testing.expectEqualDeep(RuntimeDeviceRequest{}, try parseRuntimeDeviceRequest(""));
    try std.testing.expectEqualDeep(RuntimeDeviceRequest{}, try parseRuntimeDeviceRequest("host"));
    try std.testing.expectEqualDeep(RuntimeDeviceRequest{}, try parseRuntimeDeviceRequest("cpu"));
    try std.testing.expectEqualDeep(
        RuntimeDeviceRequest{ .kind = .cuda, .cuda_device_index = 0 },
        try parseRuntimeDeviceRequest("cuda"),
    );
    try std.testing.expectEqualDeep(
        RuntimeDeviceRequest{ .kind = .cuda, .cuda_device_index = 2 },
        try parseRuntimeDeviceRequest("cuda:2"),
    );
}

test "runtime device parsing rejects invalid selectors" {
    try std.testing.expectError(error.InvalidRuntimeDeviceRequest, parseRuntimeDeviceRequest("cuda:"));
    try std.testing.expectError(error.InvalidRuntimeDeviceRequest, parseRuntimeDeviceRequest("cuda:x"));
    try std.testing.expectError(error.InvalidRuntimeDeviceRequest, parseRuntimeDeviceRequest("rocm"));
}

test "runtime device defaults to host" {
    var runtime_device = try initRuntimeDevice(null, .{});
    defer runtime_device.deinit();

    try std.testing.expect(runtime_device.isHost());
    try std.testing.expect(runtime_device.reference().is_host());
}

test "runtime device rejects cuda when support policy disallows it" {
    try std.testing.expectError(
        error.CudaUnsupported,
        initRuntimeDevice(.{ .kind = .cuda }, .{ .allow_cuda = false }),
    );
}

test "runtime device rejects cuda when the build disables it" {
    if (build_options.enable_cuda) return error.SkipZigTest;

    try std.testing.expectError(
        error.CudaNotEnabled,
        initRuntimeDevice(.{ .kind = .cuda }, .{}),
    );
}
