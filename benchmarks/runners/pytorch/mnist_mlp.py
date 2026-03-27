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


def one_hot(batch_size: int, classes: int, seed: int):
    values = [0.0] * (batch_size * classes)
    for row in range(batch_size):
        class_index = splitmix64(seed + row) % classes
        values[row * classes + class_index] = 1.0
    return values


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
        "shapes": [
            {"name": "input", "dims": spec.get("input_shape")}
        ]
        + ([{"name": "labels", "dims": spec.get("label_shape")}] if spec.get("label_shape") else []),
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

    if spec["kind"] not in {"mnist_mlp_train", "mnist_mlp_infer"}:
        print(json.dumps(make_record(spec, "skipped", "PyTorch baseline not implemented for this benchmark kind.", None)))
        return 0

    batch_size = spec["batch_size"]
    input_shape = spec["input_shape"]
    classes = 10
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

    setup_start = time.perf_counter_ns()
    model = MnistMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.tensor(
        deterministic_vector(math.prod(input_shape), seed + 11),
        dtype=torch.float32,
    ).reshape(*input_shape)
    label_tensor = None
    if spec["kind"] == "mnist_mlp_train":
        label_tensor = torch.tensor(one_hot(batch_size, classes, seed + 17), dtype=torch.float32).reshape(batch_size, classes)
    setup_latency_ns = time.perf_counter_ns() - setup_start

    warmup_iterations = spec["warmup_iterations"]
    measured_iterations = spec["measured_iterations"]
    timings_ns: list[int] = []

    def train_step():
        logits = model(inputs)
        loss = -(label_tensor * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)

    def infer_step():
        with torch.no_grad():
            _ = model(inputs)

    step = train_step if spec["kind"] == "mnist_mlp_train" else infer_step

    for _ in range(warmup_iterations):
        step()
    for _ in range(measured_iterations):
        start = time.perf_counter_ns()
        step()
        timings_ns.append(time.perf_counter_ns() - start)

    stats = summarize_timings(
        timings_ns,
        batch_size if spec["kind"].startswith("mnist_mlp") else None,
        "samples" if spec["kind"].startswith("mnist_mlp") else None,
    )
    stats["setup_latency_ns"] = setup_latency_ns
    print(json.dumps(make_record(spec, "ok", spec.get("notes", ""), stats)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
