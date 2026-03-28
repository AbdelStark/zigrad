const std = @import("std");

pub const T = f32;
pub const state_feature_count: usize = 3;
pub const action_count: usize = 3;

pub const Action = enum(u8) {
    left = 0,
    coast = 1,
    right = 2,

    pub fn fromIndex(index: usize) !Action {
        return std.meta.intToEnum(Action, @as(u8, @intCast(index))) catch error.InvalidAction;
    }

    pub fn force(self: Action) i32 {
        return switch (self) {
            .left => -1,
            .coast => 0,
            .right => 1,
        };
    }
};

pub const StartState = struct {
    position: i32,
    velocity: i32,
};

pub const CorridorConfig = struct {
    goal_position: i32 = 8,
    pit_position: i32 = 0,
    max_velocity: i32 = 1,
    max_steps: usize = 16,
    progress_reward: T = 0.12,
    step_penalty: T = -0.03,
    goal_reward: T = 1.0,
    pit_penalty: T = -1.0,

    pub fn validate(self: CorridorConfig) !void {
        if (self.goal_position <= self.pit_position + 2) return error.InvalidCorridorLength;
        if (self.max_velocity <= 0) return error.InvalidVelocityLimit;
        if (self.max_steps == 0) return error.InvalidMaxSteps;
    }

    fn corridorSpan(self: CorridorConfig) T {
        return @as(T, @floatFromInt(self.goal_position - self.pit_position));
    }
};

pub const StepResult = struct {
    state: [state_feature_count]T,
    reward: T,
    done: bool,
    reached_goal: bool,
};

pub const TransitionBatch = struct {
    states: []T,
    next_states: []T,
    actions: []usize,
    rewards: []T,
    dones: []T,
    sample_count: usize,

    pub fn deinit(self: *TransitionBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.states);
        allocator.free(self.next_states);
        allocator.free(self.actions);
        allocator.free(self.rewards);
        allocator.free(self.dones);
        self.* = undefined;
    }
};

pub const CorridorEnv = struct {
    config: CorridorConfig,
    position: i32,
    velocity: i32,
    steps: usize,
    rng: std.Random.DefaultPrng,

    pub fn init(seed: u64, config: CorridorConfig) CorridorEnv {
        config.validate() catch unreachable;
        return .{
            .config = config,
            .position = config.pit_position + 1,
            .velocity = 0,
            .steps = 0,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }

    pub fn reset(self: *CorridorEnv) [state_feature_count]T {
        const span = @as(usize, @intCast(self.config.goal_position - self.config.pit_position - 2));
        const offset = self.rng.random().uintLessThan(usize, span);
        self.position = self.config.pit_position + 1 + @as(i32, @intCast(offset));
        self.velocity = 0;
        self.steps = 0;
        return self.state();
    }

    pub fn resetTo(self: *CorridorEnv, start: StartState) ![state_feature_count]T {
        if (start.position <= self.config.pit_position or start.position >= self.config.goal_position) {
            return error.InvalidStartPosition;
        }
        if (start.velocity < -self.config.max_velocity or start.velocity > self.config.max_velocity) {
            return error.InvalidStartVelocity;
        }

        self.position = start.position;
        self.velocity = start.velocity;
        self.steps = 0;
        return self.state();
    }

    pub fn step(self: *CorridorEnv, action: Action) StepResult {
        const previous_distance = self.goalDistance();

        self.velocity = std.math.clamp(
            self.velocity + action.force(),
            -self.config.max_velocity,
            self.config.max_velocity,
        );
        self.position = std.math.clamp(
            self.position + self.velocity,
            self.config.pit_position,
            self.config.goal_position,
        );
        self.steps += 1;

        const distance_delta = previous_distance - self.goalDistance();
        var reward = self.config.step_penalty +
            (self.config.progress_reward * @as(T, @floatFromInt(distance_delta)));

        const reached_goal = self.position >= self.config.goal_position;
        const fell_into_pit = self.position <= self.config.pit_position;

        if (reached_goal) reward += self.config.goal_reward;
        if (fell_into_pit) reward += self.config.pit_penalty;

        return .{
            .state = self.state(),
            .reward = reward,
            .done = reached_goal or fell_into_pit or self.steps >= self.config.max_steps,
            .reached_goal = reached_goal,
        };
    }

    pub fn state(self: *const CorridorEnv) [state_feature_count]T {
        const span = self.config.corridorSpan();
        const position_norm = ((@as(T, @floatFromInt(self.position - self.config.pit_position)) / span) * 2.0) - 1.0;
        const velocity_norm = @as(T, @floatFromInt(self.velocity)) / @as(T, @floatFromInt(self.config.max_velocity));
        const goal_distance_norm = @as(T, @floatFromInt(self.goalDistance())) / span;
        return .{ position_norm, velocity_norm, goal_distance_norm };
    }

    fn goalDistance(self: *const CorridorEnv) i32 {
        return self.config.goal_position - self.position;
    }
};

pub fn generateSyntheticTransitionBatch(
    allocator: std.mem.Allocator,
    sample_count: usize,
    seed: u64,
    config: CorridorConfig,
) !TransitionBatch {
    try config.validate();
    if (sample_count == 0) return error.InvalidSampleCount;

    const states = try allocator.alloc(T, sample_count * state_feature_count);
    errdefer allocator.free(states);
    const next_states = try allocator.alloc(T, sample_count * state_feature_count);
    errdefer allocator.free(next_states);
    const actions = try allocator.alloc(usize, sample_count);
    errdefer allocator.free(actions);
    const rewards = try allocator.alloc(T, sample_count);
    errdefer allocator.free(rewards);
    const dones = try allocator.alloc(T, sample_count);
    errdefer allocator.free(dones);

    var env = CorridorEnv.init(seed +% 1, config);
    for (0..sample_count) |index| {
        const start = syntheticStartState(index, seed, config);
        const state = try env.resetTo(start);
        const action = syntheticAction(start.position, start.velocity, seed +% @as(u64, (index * 11) + 3), config);
        const step_result = env.step(action);

        @memcpy(states[index * state_feature_count ..][0..state_feature_count], state[0..]);
        @memcpy(next_states[index * state_feature_count ..][0..state_feature_count], step_result.state[0..]);
        actions[index] = @intFromEnum(action);
        rewards[index] = step_result.reward;
        dones[index] = if (step_result.done) 1.0 else 0.0;
    }

    return .{
        .states = states,
        .next_states = next_states,
        .actions = actions,
        .rewards = rewards,
        .dones = dones,
        .sample_count = sample_count,
    };
}

pub fn generateSyntheticStates(
    allocator: std.mem.Allocator,
    sample_count: usize,
    seed: u64,
    config: CorridorConfig,
) ![]T {
    try config.validate();
    if (sample_count == 0) return error.InvalidSampleCount;

    const states = try allocator.alloc(T, sample_count * state_feature_count);
    errdefer allocator.free(states);

    var env = CorridorEnv.init(seed +% 5, config);
    for (0..sample_count) |index| {
        const start = syntheticStartState(index, seed +% 7, config);
        const state = try env.resetTo(start);
        @memcpy(states[index * state_feature_count ..][0..state_feature_count], state[0..]);
    }

    return states;
}

fn syntheticStartState(index: usize, seed: u64, config: CorridorConfig) StartState {
    const position_span = @as(u64, @intCast(config.goal_position - config.pit_position - 1));
    const position_offset = splitmix64(seed +% @as(u64, (index * 13) + 1)) % position_span;
    const velocity_span = @as(u64, @intCast((config.max_velocity * 2) + 1));
    const raw_velocity = splitmix64(seed +% @as(u64, (index * 13) + 2)) % velocity_span;
    return .{
        .position = config.pit_position + 1 + @as(i32, @intCast(position_offset)),
        .velocity = @as(i32, @intCast(raw_velocity)) - config.max_velocity,
    };
}

fn syntheticAction(position: i32, velocity: i32, seed: u64, config: CorridorConfig) Action {
    const noise = splitmix64(seed) % 7;
    if (noise == 0 and position > config.pit_position + 1) return .left;
    if (noise == 1 and position >= config.goal_position - 2 and velocity > 0) return .coast;
    if (velocity < 0) return .right;
    if (position >= config.goal_position - 1) return .coast;
    return .right;
}

fn splitmix64(state: u64) u64 {
    var z = state +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

test "corridor env reaches the goal with repeated right pushes" {
    var env = CorridorEnv.init(7, .{});
    _ = try env.resetTo(.{ .position = 1, .velocity = 0 });

    var reached_goal = false;
    var reward_sum: T = 0;
    for (0..env.config.max_steps) |_| {
        const step_result = env.step(.right);
        reward_sum += step_result.reward;
        if (step_result.reached_goal) {
            reached_goal = true;
            break;
        }
        if (step_result.done) break;
    }

    try std.testing.expect(reached_goal);
    try std.testing.expect(reward_sum > 0);
}

test "synthetic transition batch is finite and includes terminal flags" {
    var batch = try generateSyntheticTransitionBatch(std.testing.allocator, 24, 17, .{});
    defer batch.deinit(std.testing.allocator);

    var saw_terminal = false;
    for (batch.states, batch.next_states) |state_value, next_state_value| {
        try std.testing.expect(std.math.isFinite(state_value));
        try std.testing.expect(std.math.isFinite(next_state_value));
    }
    for (batch.rewards) |reward| {
        try std.testing.expect(std.math.isFinite(reward));
    }
    for (batch.dones) |done| {
        try std.testing.expect(done == 0.0 or done == 1.0);
        saw_terminal = saw_terminal or done > 0;
    }

    try std.testing.expect(saw_terminal);
}
