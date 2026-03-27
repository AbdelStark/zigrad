# Benchmark Harness

This directory contains the RFC-0001 benchmark harness and its initial specs.

## Entry Points

From the repository root:

```sh
zig build benchmark
zig build benchmark-primitive
zig build benchmark-models
```

The default build steps write JSON-lines results into [`benchmarks/results/`](./results/).

You can also pass runtime arguments through the benchmark executable:

```sh
zig build benchmark -- --baseline pytorch
zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic.json
```

## Current Coverage

- `primitive`
  - deterministic contiguous add
  - deterministic square matmul
- `model-train`
  - synthetic MNIST-style MLP training step
- `model-infer`
  - synthetic MNIST-style MLP inference step

The MLP architecture matches the reference MNIST example dimensions while using
deterministic synthetic inputs and labels so the suite runs from a clean
checkout without dataset downloads.

## Spec Format

Benchmark specs live under [`benchmarks/specs/`](./specs/) as JSON files.

Common fields:

- `id`: stable benchmark identifier
- `suite`: `primitive`, `model-train`, or `model-infer`
- `kind`: workload selector
- `dtype`: currently `f32`
- `warmup_iterations`
- `measured_iterations`
- `thread_count`
- `seed`
- `notes`

Workload-specific fields:

- `lhs_shape`, `rhs_shape` for primitive add/matmul
- `batch_size`, `input_shape`, `label_shape` for model workloads
- `pytorch_runner` for optional baseline execution

## Output Schema

Each run emits one JSON object per benchmark with:

- benchmark id, suite, kind, runner, and status
- dtype, batch size, warmup/measured iteration counts, and seed
- shape metadata
- runtime metadata:
  - git commit
  - dirty tree flag
  - Zig version
  - harness version
  - timestamp
- system metadata:
  - operating system
  - kernel
  - CPU model
  - logical core count
  - total memory when discoverable
- backend metadata:
  - device kind
  - host provider
  - configured thread count
- setup latency
- measured latency summary:
  - min
  - median
  - mean
  - p95
  - max
  - throughput when applicable

## PyTorch Baseline

The optional PyTorch runner lives under
[`benchmarks/runners/pytorch/`](./runners/pytorch/).

If `torch` is not installed, the runner emits a `skipped` record instead of
failing the whole benchmark run. This keeps the default path dependency-light
while still allowing direct framework comparisons on prepared machines.
