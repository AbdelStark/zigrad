const std = @import("std");
const zg = @import("zigrad");
const conv_utils = zg.conv_utils;
const test_support = @import("test_support.zig");

const MnistModel = @import("examples_mnist_model").MnistModel;
const DQNModel = @import("examples_dqn_model").DQNModel;
const GCN = @import("examples_gcn_model").GCN;

fn expectTelemetry(expected: zg.device.HostOpTelemetry, actual: zg.device.HostOpTelemetry) !void {
    try std.testing.expectEqual(expected.dot_calls, actual.dot_calls);
    try std.testing.expectEqual(expected.matvec_calls, actual.matvec_calls);
    try std.testing.expectEqual(expected.matmul_calls, actual.matmul_calls);
    try std.testing.expectEqual(expected.bmm_acc_calls, actual.bmm_acc_calls);
}

fn expectDispatchTelemetry(
    expected: zg.device.HostDispatchTelemetry,
    actual: zg.device.HostDispatchTelemetry,
) !void {
    try std.testing.expectEqual(expected.direct_bmm_dispatches, actual.direct_bmm_dispatches);
    try std.testing.expectEqual(expected.fallback_bmm_dispatches, actual.fallback_bmm_dispatches);
    try std.testing.expectEqual(expected.fallback_bmm_batches, actual.fallback_bmm_batches);
}

test "host BLAS telemetry tracks direct dot matvec and batched matmul dispatch" {
    const Array = zg.NDArray(f32);

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var dot_lhs = try Array.from_slice(&.{ 1.0, 2.0, 3.0 }, &.{3}, device);
    defer dot_lhs.deinit(device);
    var dot_rhs = try Array.from_slice(&.{ 4.0, 5.0, 6.0 }, &.{3}, device);
    defer dot_rhs.deinit(device);

    var matrix = try Array.from_slice(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, &.{ 2, 3 }, device);
    defer matrix.deinit(device);
    var vector = try Array.from_slice(&.{ 1.0, -1.0, 2.0 }, &.{3}, device);
    defer vector.deinit(device);

    var rhs = try Array.from_slice(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }, &.{ 3, 2 }, device);
    defer rhs.deinit(device);

    host.resetOpTelemetry();

    var dot_out = try dot_lhs.dot(dot_rhs, device);
    defer dot_out.deinit(device);
    var matvec_out = try matrix.matvec(vector, device, .{});
    defer matvec_out.deinit(device);
    var matmul_out = try matrix.bmm(rhs, device, .{});
    defer matmul_out.deinit(device);

    const telemetry = host.opTelemetry();
    try expectTelemetry(.{
        .dot_calls = 1,
        .matvec_calls = 1,
        .matmul_calls = 1,
        .bmm_acc_calls = 1,
    }, telemetry);
    try std.testing.expectEqual(@as(u64, 3), telemetry.totalBlasCalls());
    try expectDispatchTelemetry(.{
        .direct_bmm_dispatches = 1,
    }, host.dispatchTelemetry());

    host.resetOpTelemetry();
    try expectTelemetry(.{}, host.opTelemetry());
    host.resetDispatchTelemetry();
    try expectDispatchTelemetry(.{}, host.dispatchTelemetry());
}

test "mnist example forward uses three host batched matmul dispatches" {
    const Tensor = zg.NDTensor(f32);
    const allocator = std.testing.allocator;
    const input_shape = [_]usize{ 2, 28, 28 };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var model = try MnistModel(f32).initWithGraph(device, &graph);
    defer model.deinit();

    const input_values = try test_support.makeDeterministicSlice(
        allocator,
        test_support.countElements(&input_shape),
        11,
    );
    defer allocator.free(input_values);

    const input = try Tensor.from_slice(device, input_values, &input_shape, .{ .graph = &graph });
    defer input.deinit();

    host.resetOpTelemetry();
    const previous_grad_enabled = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_enabled;
    const output = try model.forward(input);
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 2, 10 }, output.get_shape());
    try expectTelemetry(.{
        .matmul_calls = 3,
        .bmm_acc_calls = 3,
    }, host.opTelemetry());
    try expectDispatchTelemetry(.{
        .direct_bmm_dispatches = 3,
    }, host.dispatchTelemetry());
}

test "dqn example forward uses three host batched matmul dispatches" {
    const Tensor = zg.NDTensor(f32);
    const allocator = std.testing.allocator;
    const input_shape = [_]usize{ 3, 4 };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var model = try DQNModel(f32, 3).initWithGraph(device, 4, 128, 2, &graph);
    defer model.deinit();

    const input_values = try test_support.makeDeterministicSlice(
        allocator,
        test_support.countElements(&input_shape),
        29,
    );
    defer allocator.free(input_values);

    const input = try Tensor.from_slice(device, input_values, &input_shape, .{ .graph = &graph });
    defer input.deinit();

    host.resetOpTelemetry();
    const previous_grad_enabled = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_enabled;
    const output = try model.forward(input);
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, output.get_shape());
    try expectTelemetry(.{
        .matmul_calls = 3,
        .bmm_acc_calls = 3,
    }, host.opTelemetry());
    try expectDispatchTelemetry(.{
        .direct_bmm_dispatches = 3,
    }, host.dispatchTelemetry());
}

test "gcn example forward uses two host batched matmul dispatches" {
    const Tensor = zg.NDTensor(f32);
    const allocator = std.testing.allocator;
    const input_shape = [_]usize{ 5, 3 };

    var graph = zg.Graph.init(allocator, .{ .eager_teardown = true });
    defer graph.deinit();

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var model = try GCN(f32).init(device, input_shape[1], 2, .{ .graph = &graph });
    defer model.deinit();

    const input_values = try test_support.makeDeterministicSlice(
        allocator,
        test_support.countElements(&input_shape),
        47,
    );
    defer allocator.free(input_values);
    const edge_values = try test_support.makeGraphEdgeIndex(allocator, input_shape[0], 4);
    defer allocator.free(edge_values);

    const input = try Tensor.from_slice(device, input_values, &input_shape, .{ .graph = &graph });
    defer input.deinit();
    const edge_index = try zg.NDTensor(usize).from_slice(device, edge_values, &.{ 2, edge_values.len / 2 }, .{
        .graph = &graph,
    });
    defer edge_index.deinit();

    host.resetOpTelemetry();
    const previous_grad_enabled = zg.runtime.grad_enabled;
    zg.runtime.grad_enabled = false;
    defer zg.runtime.grad_enabled = previous_grad_enabled;
    const output = try model.forward(input, edge_index);
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &.{ input_shape[0], 2 }, output.get_shape());
    try expectTelemetry(.{
        .matmul_calls = 2,
        .bmm_acc_calls = 2,
    }, host.opTelemetry());
    try expectDispatchTelemetry(.{
        .direct_bmm_dispatches = 2,
    }, host.dispatchTelemetry());
}

test "legacy conv2d im2col path uses one batched dispatch across the batch" {
    const Array = zg.NDArray(f32);

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    const input_shape = [_]usize{ 2, 1, 4, 4 };
    const weight_shape = [_]usize{ 2, 1, 2, 2 };

    const input_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&input_shape),
        71,
    );
    defer std.testing.allocator.free(input_values);
    const weight_values = try test_support.makeDeterministicSlice(
        std.testing.allocator,
        test_support.countElements(&weight_shape),
        73,
    );
    defer std.testing.allocator.free(weight_values);

    var input = try Array.from_slice(input_values, &input_shape, device);
    defer input.deinit(device);
    var weights = try Array.from_slice(weight_values, &weight_shape, device);
    defer weights.deinit(device);

    host.resetOpTelemetry();
    var output = try conv_utils.conv2dForwardIm2col(f32, input, weights, null, .{}, device);
    defer output.deinit(device);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3, 3 }, output.shape.slice());
    try expectTelemetry(.{
        .matmul_calls = 2,
        .bmm_acc_calls = 1,
    }, host.opTelemetry());
    try expectDispatchTelemetry(.{
        .direct_bmm_dispatches = 1,
    }, host.dispatchTelemetry());
}

test "nested broadcast matmul reports fallback dispatch telemetry" {
    const Array = zg.NDArray(f32);

    var host = zg.device.HostDevice.init();
    defer host.deinit();
    const device = host.reference();

    var lhs = try Array.from_slice(
        &.{
            1,  2,  3,  4,  5,  6,
            7,  8,  9,  10, 11, 12,
            13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24,
        },
        &.{ 2, 2, 2, 3 },
        device,
    );
    defer lhs.deinit(device);
    var rhs = try Array.from_slice(
        &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 },
        &.{ 2, 1, 3, 2 },
        device,
    );
    defer rhs.deinit(device);

    host.resetOpTelemetry();
    host.resetDispatchTelemetry();
    var output = try lhs.bmm(rhs, device, .{});
    defer output.deinit(device);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, output.shape.slice());
    try expectTelemetry(.{
        .matmul_calls = 4,
    }, host.opTelemetry());
    try expectDispatchTelemetry(.{
        .fallback_bmm_dispatches = 1,
        .fallback_bmm_batches = 4,
    }, host.dispatchTelemetry());
}
