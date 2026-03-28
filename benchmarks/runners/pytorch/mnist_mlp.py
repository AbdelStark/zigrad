#!/usr/bin/env python3
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def splitmix64(state: int) -> int:
    z = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def deterministic_vector(count: int, seed: int):
    values = []
    for index in range(count):
        mixed = splitmix64(seed + index)
        normalized = (mixed % 5000) / 5000.0 - 0.5
        values.append(normalized)
    return values


def deterministic_indices(count: int, modulus: int, seed: int):
    values = []
    for index in range(count):
        values.append(splitmix64(seed + index) % modulus)
    return values


def one_hot(batch_size: int, classes: int, seed: int):
    values = [0.0] * (batch_size * classes)
    for row in range(batch_size):
        class_index = splitmix64(seed + row) % classes
        values[row * classes + class_index] = 1.0
    return values


def reward_values(batch_size: int, seed: int):
    return [value * 0.5 for value in deterministic_vector(batch_size, seed)]


def done_values(batch_size: int, seed: int):
    values = []
    for row in range(batch_size):
        values.append(1.0 if splitmix64(seed + row) % 7 == 0 else 0.0)
    return values


def ring_skip_edges(node_count: int, fanout: int = 4):
    src = []
    tgt = []
    for node in range(node_count):
        src.extend([node] * fanout)
        tgt.append(node)
        tgt.append(node_count - 1 if node == 0 else node - 1)
        tgt.append((node + 1) % node_count)
        tgt.append((node + 2) % node_count)
    return src, tgt


def conv_output_shape(spec: dict):
    input_shape = spec["lhs_shape"]
    weight_shape = spec["rhs_shape"]
    stride = spec.get("stride", 1)
    padding = spec.get("padding", 0)
    dilation = spec.get("dilation", 1)
    kernel_size = weight_shape[2]
    effective_kernel = dilation * (kernel_size - 1) + 1
    output_height = ((input_shape[2] + 2 * padding - effective_kernel) // stride) + 1
    output_width = ((input_shape[3] + 2 * padding - effective_kernel) // stride) + 1
    return [input_shape[0], weight_shape[0], output_height, output_width]


def host_provider() -> str:
    if sys.platform == "darwin":
        return "accelerate"
    return "cpu"


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        )
    except Exception:
        return True


def cpu_model() -> str:
    if sys.platform == "darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                return subprocess.check_output(["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL).strip()
            except Exception:
                continue
    return platform.processor() or platform.machine()


def shape_metadata(spec: dict):
    kind = spec["kind"]
    if kind in {"blas_dot", "autograd_dot_backward"}:
        return [
            {"name": "lhs", "dims": spec["lhs_shape"]},
            {"name": "rhs", "dims": spec["rhs_shape"]},
        ]

    if kind in {"blas_matvec", "autograd_matvec_backward"}:
        return [
            {"name": "matrix", "dims": spec["lhs_shape"]},
            {"name": "vector", "dims": spec["rhs_shape"]},
        ]

    if kind == "blas_conv2d_im2col":
        return [
            {"name": "input", "dims": spec["lhs_shape"]},
            {"name": "weights", "dims": spec["rhs_shape"]},
            {"name": "output", "dims": conv_output_shape(spec)},
        ]

    input_shape = spec.get("input_shape")
    batch_size = spec.get("batch_size")

    if kind in {"mnist_mlp_train", "mnist_mlp_infer"}:
        shapes = [{"name": "input", "dims": input_shape}]
        if spec.get("label_shape"):
            shapes.append({"name": "labels", "dims": spec["label_shape"]})
        return shapes

    if kind == "dqn_cartpole_train":
        return [
            {"name": "state", "dims": input_shape},
            {"name": "next_state", "dims": input_shape},
            {"name": "action", "dims": [batch_size, 1]},
            {"name": "reward", "dims": [batch_size, 1]},
            {"name": "done", "dims": [batch_size, 1]},
        ]

    if kind == "dqn_cartpole_infer":
        return [{"name": "state", "dims": input_shape}]

    node_count = input_shape[0]
    edge_count = node_count * 4
    shapes = [
        {"name": "node_features", "dims": input_shape},
        {"name": "edge_index", "dims": [2, edge_count]},
    ]
    if kind == "gcn_train":
        shapes.append({"name": "labels", "dims": spec["label_shape"]})
    return shapes


def make_record(spec: dict, status: str, notes: str, stats: dict | None):
    return {
        "benchmark_id": spec["id"],
        "suite": spec["suite"],
        "kind": spec["kind"],
        "runner": "pytorch",
        "status": status,
        "dtype": spec.get("dtype", "f32"),
        "warmup_iterations": spec.get("warmup_iterations", 0),
        "measured_iterations": spec.get("measured_iterations", 0),
        "batch_size": spec.get("batch_size"),
        "seed": spec.get("seed", 81761),
        "shapes": shape_metadata(spec),
        "runtime": {
            "timestamp_unix_ms": int(time.time() * 1000),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "zig_version": "n/a",
            "harness_version": "0.1.0",
        },
        "system": {
            "os": platform.system().lower(),
            "kernel": platform.release(),
            "arch": platform.machine(),
            "cpu_model": cpu_model(),
            "cpu_logical_cores": os.cpu_count() or 0,
            "total_memory_bytes": None,
        },
        "backend": {
            "device": "cpu",
            "host_provider": host_provider(),
            "thread_count": spec.get("thread_count"),
            "accelerator": None,
        },
        "setup_latency_ns": None if stats is None else stats.pop("setup_latency_ns"),
        "stats": stats,
        "notes": notes,
    }


def summarize_timings(timings_ns: list[int], throughput_items: int | None, throughput_unit: str | None):
    sorted_timings = sorted(timings_ns)
    mean_ns = statistics.fmean(sorted_timings)
    p95_index = 0 if len(sorted_timings) == 1 else ((len(sorted_timings) - 1) * 95) // 100
    throughput = None
    if throughput_items is not None and mean_ns > 0:
        throughput = throughput_items / (mean_ns / 1_000_000_000.0)
    return {
        "min_ns": sorted_timings[0],
        "median_ns": int(statistics.median(sorted_timings)),
        "mean_ns": mean_ns,
        "p95_ns": sorted_timings[p95_index],
        "max_ns": sorted_timings[-1],
        "throughput_per_second": throughput,
        "throughput_unit": throughput_unit if throughput is not None else None,
    }


def throughput_shape(spec: dict):
    kind = spec["kind"]
    if kind in {"blas_dot", "autograd_dot_backward"}:
        return spec["lhs_shape"][0], "elements"
    if kind in {"blas_matvec", "autograd_matvec_backward"}:
        return math.prod(spec["lhs_shape"]), "matrix-elements"
    if kind == "blas_conv2d_im2col":
        return spec["lhs_shape"][0], "samples"
    if kind in {"mnist_mlp_train", "mnist_mlp_infer", "dqn_cartpole_train", "dqn_cartpole_infer"}:
        return spec["batch_size"], "samples"
    if kind in {"gcn_train", "gcn_infer"}:
        return spec["input_shape"][0], "nodes"
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        print(json.dumps(make_record(spec, "skipped", f"PyTorch unavailable: {exc}", None)))
        return 0

    kind = spec["kind"]
    supported_kinds = {
        "blas_dot",
        "blas_matvec",
        "blas_conv2d_im2col",
        "autograd_dot_backward",
        "autograd_matvec_backward",
        "mnist_mlp_train",
        "mnist_mlp_infer",
        "dqn_cartpole_train",
        "dqn_cartpole_infer",
        "gcn_train",
        "gcn_infer",
    }
    if kind not in supported_kinds:
        print(json.dumps(make_record(spec, "skipped", "PyTorch baseline not implemented for this benchmark kind.", None)))
        return 0

    seed = spec.get("seed", 81761)
    thread_count = spec.get("thread_count")
    if thread_count:
        torch.set_num_threads(thread_count)

    def linear_weight(out_features: int, in_features: int, layer_seed: int):
        values = deterministic_vector(out_features * in_features, layer_seed)
        tensor = torch.tensor(values, dtype=torch.float32).reshape(out_features, in_features)
        return tensor * 0.1

    class MnistMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w0 = torch.nn.Parameter(linear_weight(128, 784, seed + 1))
            self.b0 = torch.nn.Parameter(torch.zeros(128, dtype=torch.float32))
            self.w1 = torch.nn.Parameter(linear_weight(64, 128, seed + 2))
            self.b1 = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))
            self.w2 = torch.nn.Parameter(linear_weight(10, 64, seed + 3))
            self.b2 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float32))

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            x = F.relu(F.linear(x, self.w0, self.b0))
            x = F.relu(F.linear(x, self.w1, self.b1))
            return F.linear(x, self.w2, self.b2)

    class DQNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w0 = torch.nn.Parameter(linear_weight(128, 4, seed + 1))
            self.b0 = torch.nn.Parameter(torch.zeros(128, dtype=torch.float32))
            self.w1 = torch.nn.Parameter(linear_weight(128, 128, seed + 2))
            self.b1 = torch.nn.Parameter(torch.zeros(128, dtype=torch.float32))
            self.w2 = torch.nn.Parameter(linear_weight(2, 128, seed + 3))
            self.b2 = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, x):
            x = x.reshape(x.shape[0], -1)
            x = F.relu(F.linear(x, self.w0, self.b0))
            x = F.relu(F.linear(x, self.w1, self.b1))
            return F.linear(x, self.w2, self.b2)

    class GCNModel(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.w0 = torch.nn.Parameter(linear_weight(16, in_features, seed + 1))
            self.b0 = torch.nn.Parameter(torch.zeros(16, dtype=torch.float32))
            self.w1 = torch.nn.Parameter(linear_weight(out_features, 16, seed + 2))
            self.b1 = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

        @staticmethod
        def propagate(h, edge_index):
            src = edge_index[0]
            tgt = edge_index[1]
            node_count = h.shape[0]
            deg = torch.ones(node_count, dtype=h.dtype)
            deg.index_add_(0, tgt, torch.ones_like(tgt, dtype=h.dtype))
            deg_norm = deg.rsqrt()

            src_scale = deg_norm.index_select(0, src).unsqueeze(-1)
            tgt_scale = deg_norm.index_select(0, tgt).unsqueeze(-1)
            messages = h.index_select(0, src) * src_scale * tgt_scale

            out = torch.zeros((node_count, h.shape[1]), dtype=h.dtype)
            out.index_add_(0, tgt, messages)
            return out

        def layer(self, x, edge_index, weight, bias):
            h = F.linear(x, weight, None)
            return self.propagate(h, edge_index) + bias

        def forward(self, x, edge_index):
            x = F.relu(self.layer(x, edge_index, self.w0, self.b0))
            return self.layer(x, edge_index, self.w1, self.b1)

    def make_leaf_tensor(values, shape):
        return torch.tensor(values, dtype=torch.float32).reshape(*shape).clone().detach().requires_grad_(True)

    setup_start = time.perf_counter_ns()
    step = None

    if kind in {"blas_dot", "blas_matvec", "blas_conv2d_im2col", "autograd_dot_backward", "autograd_matvec_backward"}:
        lhs_shape = spec["lhs_shape"]
        rhs_shape = spec["rhs_shape"]
        lhs_values = deterministic_vector(math.prod(lhs_shape), seed)
        rhs_values = deterministic_vector(math.prod(rhs_shape), seed + 1)

        if kind == "blas_dot":
            lhs = torch.tensor(lhs_values, dtype=torch.float32).reshape(*lhs_shape)
            rhs = torch.tensor(rhs_values, dtype=torch.float32).reshape(*rhs_shape)

            def dot_step():
                with torch.no_grad():
                    _ = torch.dot(lhs, rhs)

            step = dot_step

        elif kind == "blas_matvec":
            matrix = torch.tensor(lhs_values, dtype=torch.float32).reshape(*lhs_shape)
            vector = torch.tensor(rhs_values, dtype=torch.float32).reshape(*rhs_shape)

            def matvec_step():
                with torch.no_grad():
                    _ = torch.mv(matrix, vector)

            step = matvec_step

        elif kind == "blas_conv2d_im2col":
            stride = spec.get("stride", 1)
            padding = spec.get("padding", 0)
            dilation = spec.get("dilation", 1)
            inputs = torch.tensor(lhs_values, dtype=torch.float32).reshape(*lhs_shape)
            weights = torch.tensor(rhs_values, dtype=torch.float32).reshape(*rhs_shape)

            def conv_step():
                with torch.no_grad():
                    _ = F.conv2d(inputs, weights, bias=None, stride=stride, padding=padding, dilation=dilation)

            step = conv_step

        elif kind == "autograd_dot_backward":
            lhs = make_leaf_tensor(lhs_values, lhs_shape)
            rhs = make_leaf_tensor(rhs_values, rhs_shape)

            def autograd_dot_step():
                if lhs.grad is not None:
                    lhs.grad.zero_()
                if rhs.grad is not None:
                    rhs.grad.zero_()
                output = torch.dot(lhs, rhs)
                output.backward()

            step = autograd_dot_step

        else:
            matrix = make_leaf_tensor(lhs_values, lhs_shape)
            vector = make_leaf_tensor(rhs_values, rhs_shape)

            def autograd_matvec_step():
                if matrix.grad is not None:
                    matrix.grad.zero_()
                if vector.grad is not None:
                    vector.grad.zero_()
                output = torch.mv(matrix, vector)
                output.backward(torch.ones_like(output))

            step = autograd_matvec_step

    elif kind in {"mnist_mlp_train", "mnist_mlp_infer"}:
        batch_size = spec["batch_size"]
        input_shape = spec["input_shape"]
        model = MnistMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        inputs = torch.tensor(
            deterministic_vector(math.prod(input_shape), seed + 11),
            dtype=torch.float32,
        ).reshape(*input_shape)
        labels = None
        if kind == "mnist_mlp_train":
            labels = torch.tensor(one_hot(batch_size, 10, seed + 17), dtype=torch.float32).reshape(batch_size, 10)

        def train_step():
            logits = model(inputs)
            loss = -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)

        def infer_step():
            with torch.no_grad():
                _ = model(inputs)

        step = train_step if kind == "mnist_mlp_train" else infer_step

    elif kind in {"dqn_cartpole_train", "dqn_cartpole_infer"}:
        batch_size = spec["batch_size"]
        input_shape = spec["input_shape"]
        policy = DQNModel()
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        states = torch.tensor(
            deterministic_vector(math.prod(input_shape), seed + 31),
            dtype=torch.float32,
        ).reshape(*input_shape)

        if kind == "dqn_cartpole_train":
            target = DQNModel()
            target.load_state_dict(policy.state_dict())
            next_states = torch.tensor(
                deterministic_vector(math.prod(input_shape), seed + 37),
                dtype=torch.float32,
            ).reshape(*input_shape)
            actions = torch.tensor(deterministic_indices(batch_size, 2, seed + 41), dtype=torch.int64).reshape(batch_size, 1)
            rewards = torch.tensor(reward_values(batch_size, seed + 43), dtype=torch.float32).reshape(batch_size, 1)
            dones = torch.tensor(done_values(batch_size, seed + 47), dtype=torch.float32).reshape(batch_size, 1)

            def train_step():
                with torch.no_grad():
                    next_q = target(next_states)
                    max_next_q = next_q.max(dim=1, keepdim=True).values
                    targets = rewards + 0.99 * max_next_q * (1.0 - dones)

                all_q = policy(states)
                selected_q = all_q.gather(1, actions)
                loss = F.smooth_l1_loss(selected_q, targets, reduction="mean")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)

            step = train_step
        else:

            def infer_step():
                with torch.no_grad():
                    _ = policy(states)

            step = infer_step

    else:
        input_shape = spec["input_shape"]
        node_count = input_shape[0]
        feature_count = input_shape[1]
        output_features = spec.get("label_shape", [node_count, 7])[1]
        src, tgt = ring_skip_edges(node_count)
        edge_index = torch.tensor([src, tgt], dtype=torch.int64)
        inputs = torch.tensor(
            deterministic_vector(math.prod(input_shape), seed + 59),
            dtype=torch.float32,
        ).reshape(*input_shape)
        model = GCNModel(feature_count, output_features)
        if kind == "gcn_train":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-8)
            labels = torch.tensor(one_hot(node_count, output_features, seed + 61), dtype=torch.float32).reshape(
                node_count, output_features
            )

            def train_step():
                logits = model(inputs, edge_index)
                loss = -(labels * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=False)

            step = train_step
        else:

            def infer_step():
                with torch.no_grad():
                    _ = model(inputs, edge_index)

            step = infer_step

    setup_latency_ns = time.perf_counter_ns() - setup_start

    warmup_iterations = spec["warmup_iterations"]
    measured_iterations = spec["measured_iterations"]
    timings_ns: list[int] = []

    for _ in range(warmup_iterations):
        step()
    for _ in range(measured_iterations):
        start = time.perf_counter_ns()
        step()
        timings_ns.append(time.perf_counter_ns() - start)

    throughput_items, throughput_unit = throughput_shape(spec)
    stats = summarize_timings(timings_ns, throughput_items, throughput_unit)
    stats["setup_latency_ns"] = setup_latency_ns
    print(json.dumps(make_record(spec, "ok", spec.get("notes", ""), stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
