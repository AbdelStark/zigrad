const std = @import("std");
const zg = @import("zigrad");

pub const T = f32;
pub const input_feature_count: usize = 4;
pub const output_feature_count: usize = 3;

const Tensor = zg.NDTensor(T);
const pi = @as(T, @floatCast(std.math.pi));
const two_pi = 2 * pi;

pub const DynamicsConfig = struct {
    gravity: T = 9.81,
    mass: T = 1.0,
    length: T = 1.0,
    damping: T = 0.08,
    dt: T = 0.05,
    max_speed: T = 8.0,
    max_torque: T = 2.0,

    pub fn simulateStep(self: DynamicsConfig, state: [2]T, applied_torque: T) [2]T {
        const torque = std.math.clamp(applied_torque, -self.max_torque, self.max_torque);
        const inertia = self.mass * self.length * self.length;
        const angular_acc = -(@as(T, self.gravity) / self.length) * @sin(state[0]) -
            (self.damping * state[1]) +
            (torque / inertia);
        const next_omega = std.math.clamp(
            state[1] + (self.dt * angular_acc),
            -self.max_speed,
            self.max_speed,
        );
        const next_theta = wrapAngle(state[0] + (self.dt * next_omega));
        return .{ next_theta, next_omega };
    }

    pub fn normalizeInput(self: DynamicsConfig, state: [2]T, torque: T) [input_feature_count]T {
        return .{
            @sin(state[0]),
            @cos(state[0]),
            std.math.clamp(state[1] / self.max_speed, -1, 1),
            std.math.clamp(torque / self.max_torque, -1, 1),
        };
    }

    pub fn normalizeState(self: DynamicsConfig, state: [2]T) [output_feature_count]T {
        return .{
            @sin(state[0]),
            @cos(state[0]),
            std.math.clamp(state[1] / self.max_speed, -1, 1),
        };
    }

    pub fn denormalizeState(self: DynamicsConfig, normalized_state: [output_feature_count]T) [2]T {
        return .{
            wrapAngle(std.math.atan2(normalized_state[0], normalized_state[1])),
            std.math.clamp(normalized_state[2], -1, 1) * self.max_speed,
        };
    }
};

pub const GeneratedTransitions = struct {
    inputs: []T,
    targets: []T,
    sample_count: usize,

    pub fn deinit(self: *GeneratedTransitions, allocator: std.mem.Allocator) void {
        allocator.free(self.inputs);
        allocator.free(self.targets);
        self.* = undefined;
    }
};

pub const PendulumDataset = struct {
    allocator: std.mem.Allocator,
    dynamics: DynamicsConfig,
    inputs: []T,
    targets: []T,
    batch_size: usize,
    sample_count: usize,

    pub const Batch = struct {
        inputs: *Tensor,
        targets: *Tensor,
        size: usize,

        pub fn deinit(self: *Batch) void {
            self.inputs.deinit();
            self.targets.deinit();
            self.* = undefined;
        }
    };

    pub fn initGenerated(
        allocator: std.mem.Allocator,
        dynamics: DynamicsConfig,
        sample_count: usize,
        batch_size: usize,
        seed: u64,
    ) !PendulumDataset {
        if (sample_count == 0) return error.InvalidSampleCount;
        if (batch_size == 0) return error.InvalidBatchSize;

        const generated = try generateTransitions(allocator, sample_count, seed, dynamics);
        return .{
            .allocator = allocator,
            .dynamics = dynamics,
            .inputs = generated.inputs,
            .targets = generated.targets,
            .batch_size = batch_size,
            .sample_count = generated.sample_count,
        };
    }

    pub fn deinit(self: *PendulumDataset) void {
        self.allocator.free(self.inputs);
        self.allocator.free(self.targets);
        self.* = undefined;
    }

    pub fn batchCount(self: *const PendulumDataset) usize {
        return (self.sample_count + self.batch_size - 1) / self.batch_size;
    }

    pub fn makeBatch(
        self: *const PendulumDataset,
        device: zg.DeviceReference,
        batch_index: usize,
        graph: ?*zg.Graph,
    ) !Batch {
        const start = batch_index * self.batch_size;
        if (start >= self.sample_count) return error.InvalidBatchIndex;

        const actual_batch_size = @min(self.batch_size, self.sample_count - start);
        const input_offset = start * input_feature_count;
        const target_offset = start * output_feature_count;

        const inputs = try Tensor.from_slice(
            device,
            self.inputs[input_offset .. input_offset + (actual_batch_size * input_feature_count)],
            &.{ actual_batch_size, input_feature_count },
            .{ .graph = graph },
        );
        errdefer inputs.deinit();

        const targets = try Tensor.from_slice(
            device,
            self.targets[target_offset .. target_offset + (actual_batch_size * output_feature_count)],
            &.{ actual_batch_size, output_feature_count },
            .{ .graph = graph },
        );

        return .{
            .inputs = inputs,
            .targets = targets,
            .size = actual_batch_size,
        };
    }
};

pub fn generateTransitions(
    allocator: std.mem.Allocator,
    sample_count: usize,
    seed: u64,
    dynamics: DynamicsConfig,
) !GeneratedTransitions {
    if (sample_count == 0) return error.InvalidSampleCount;

    const inputs = try allocator.alloc(T, sample_count * input_feature_count);
    errdefer allocator.free(inputs);
    const targets = try allocator.alloc(T, sample_count * output_feature_count);
    errdefer allocator.free(targets);

    for (0..sample_count) |sample_index| {
        const sample_seed = seed +% (@as(u64, sample_index) *% 17);
        const state = sampledState(dynamics, sample_seed);
        const torque = sampledTorque(dynamics, sample_seed +% 3);
        const next_state = dynamics.simulateStep(state, torque);
        const input = dynamics.normalizeInput(state, torque);
        const target = dynamics.normalizeState(next_state);

        @memcpy(
            inputs[sample_index * input_feature_count ..][0..input_feature_count],
            input[0..],
        );
        @memcpy(
            targets[sample_index * output_feature_count ..][0..output_feature_count],
            target[0..],
        );
    }

    return .{
        .inputs = inputs,
        .targets = targets,
        .sample_count = sample_count,
    };
}

pub fn deterministicInitialState(dynamics: DynamicsConfig, seed: u64) [2]T {
    return .{
        signedUnit(seed +% 1) * (pi * 0.55),
        signedUnit(seed +% 2) * (dynamics.max_speed * 0.35),
    };
}

pub fn makeRolloutTorques(
    allocator: std.mem.Allocator,
    dynamics: DynamicsConfig,
    steps: usize,
    seed: u64,
) ![]T {
    const torques = try allocator.alloc(T, steps);
    for (torques, 0..) |*torque, step_index| {
        const smooth_component = @as(T, 0.35) * @sin(@as(T, @floatFromInt(step_index)) * 0.4);
        const random_component = @as(T, 0.5) * signedUnit(seed +% @as(u64, (step_index * 5) + 1));
        torque.* = std.math.clamp(
            (smooth_component + random_component) * dynamics.max_torque,
            -dynamics.max_torque,
            dynamics.max_torque,
        );
    }
    return torques;
}

pub fn angleDifference(lhs: T, rhs: T) T {
    return wrapAngle(lhs - rhs);
}

fn sampledState(dynamics: DynamicsConfig, seed: u64) [2]T {
    return .{
        signedUnit(seed +% 1) * pi,
        signedUnit(seed +% 2) * (dynamics.max_speed * 0.8),
    };
}

fn sampledTorque(dynamics: DynamicsConfig, seed: u64) T {
    return signedUnit(seed +% 5) * (dynamics.max_torque * 0.95);
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn signedUnit(seed: u64) T {
    const mixed = splitmix64(seed);
    const normalized = (@as(f64, @floatFromInt(mixed % 20_000)) / 10_000.0) - 1.0;
    return @as(T, @floatCast(normalized));
}

fn wrapAngle(theta: T) T {
    var wrapped = theta;
    while (wrapped > pi) wrapped -= two_pi;
    while (wrapped < -pi) wrapped += two_pi;
    return wrapped;
}

test "generated pendulum transitions stay finite and normalized" {
    var generated = try generateTransitions(std.testing.allocator, 8, 7, .{});
    defer generated.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 8 * input_feature_count), generated.inputs.len);
    try std.testing.expectEqual(@as(usize, 8 * output_feature_count), generated.targets.len);

    for (generated.inputs) |value| {
        try std.testing.expect(std.math.isFinite(value));
        try std.testing.expect(value >= -1.0001 and value <= 1.0001);
    }
    for (generated.targets) |value| {
        try std.testing.expect(std.math.isFinite(value));
        try std.testing.expect(value >= -1.0001 and value <= 1.0001);
    }
}
