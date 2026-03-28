#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


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
    if kind in {"mnist_mlp_train", "mnist_mlp_infer"}:
        shapes = [{"name": "input", "dims": spec["input_shape"]}]
        if spec.get("label_shape"):
            shapes.append({"name": "labels", "dims": spec["label_shape"]})
        return shapes
    return [{"name": "input", "dims": spec.get("input_shape") or spec.get("lhs_shape") or []}]


def throughput(spec: dict):
    kind = spec["kind"]
    if kind in {"blas_dot", "autograd_dot_backward"}:
        return spec["lhs_shape"][0], "elements"
    if kind in {"mnist_mlp_train", "mnist_mlp_infer"}:
        return spec.get("batch_size"), "samples"
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--thread-count", type=int)
    args = parser.parse_args()

    mode = os.getenv("ZIGRAD_BASELINE_SMOKE_MODE", "ok")
    if mode == "invalid-json":
        print("not json")
        return 0
    if mode == "exit-7":
        print("fixture runner forced exit", file=sys.stderr)
        return 7
    if mode == "empty":
        return 0

    spec = json.loads(Path(args.spec).read_text())
    if args.thread_count is not None:
        spec["thread_count"] = args.thread_count

    throughput_value, throughput_unit = throughput(spec)
    record = {
        "benchmark_id": spec["id"],
        "spec_path": args.spec,
        "suite": spec["suite"],
        "kind": spec["kind"],
        "runner": "pytorch",
        "status": "ok",
        "dtype": spec.get("dtype", "f32"),
        "warmup_iterations": spec["warmup_iterations"],
        "measured_iterations": spec["measured_iterations"],
        "batch_size": spec.get("batch_size"),
        "seed": spec.get("seed", 81761),
        "shapes": shape_metadata(spec),
        "provenance": spec.get("provenance"),
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
            "cpu_logical_cores": os.cpu_count() or 1,
            "cpu_frequency_policy": None,
            "total_memory_bytes": None,
        },
        "backend": {
            "device": "cpu",
            "host_provider": host_provider(),
            "thread_count": spec.get("thread_count"),
            "accelerator": None,
        },
        "setup_latency_ns": 1000,
        "stats": {
            "min_ns": 900,
            "median_ns": 1000,
            "mean_ns": 1000.0,
            "p95_ns": 1100,
            "max_ns": 1200,
            "throughput_per_second": None if throughput_value is None else float(throughput_value) * 1_000_000.0,
            "throughput_unit": throughput_unit,
        },
        "notes": "synthetic baseline fixture output",
    }
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
