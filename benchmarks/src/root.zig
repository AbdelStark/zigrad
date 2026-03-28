pub const cli = @import("cli.zig");
pub const compare = @import("compare.zig");
pub const manifest = @import("manifest.zig");
pub const metadata = @import("metadata.zig");
pub const result = @import("result.zig");
pub const workload = @import("workload.zig");

test {
    _ = @import("compare.zig");
    _ = @import("dqn_bench_model.zig");
    _ = @import("gcn_bench_model.zig");
    _ = @import("manifest.zig");
    _ = @import("provider_audit.zig");
    _ = @import("provider_parity.zig");
    _ = @import("result.zig");
    _ = @import("test_support.zig");
    _ = @import("workload.zig");
}
