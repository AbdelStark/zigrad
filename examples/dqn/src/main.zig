const std = @import("std");
const std_options = .{ .log_level = .info };

pub fn main() !void {
    const train = @import("dqn_train.zig");
    if ((std.posix.getenv("ZG_EXAMPLE_SMOKE") orelse "0")[0] == '1') {
        _ = try train.trainDQNSmoke();
        return;
    }
    try train.trainDQN();
}
