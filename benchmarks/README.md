# Benchmark Harness

This directory contains the RFC-0001 benchmark harness and its current
synthetic smoke-scale specs.

## Entry Points

From the repository root:

```sh
zig build benchmark
zig build test-provider-parity
zig build benchmark-primitive
zig build benchmark-blas
zig build benchmark-autograd
zig build benchmark-memory
zig build benchmark-models
zig build benchmark-compare -- --baseline benchmarks/results/baseline.jsonl --candidate benchmarks/results/latest.jsonl
```

Host BLAS selection follows the main build graph:

```sh
zig build benchmark -Dhost_blas=openblas
zig build benchmark -Dhost_blas=mkl -Dmkl_include_dir=/opt/intel/oneapi/mkl/latest/include -Dmkl_library_dir=/opt/intel/oneapi/mkl/latest/lib
zig build test-provider-parity -Dhost_blas=openblas
zig build test-provider-parity -Dhost_blas=mkl -Dmkl_include_dir=/opt/intel/oneapi/mkl/latest/include -Dmkl_library_dir=/opt/intel/oneapi/mkl/latest/lib
```

The default build steps write JSON-lines results into [`benchmarks/results/`](./results/).

You can also pass runtime arguments through the benchmark executable:

```sh
zig build benchmark -- --baseline pytorch
zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic.json
```

The comparison utility reads two JSONL files and classifies regressions using
the RFC-0001 default policy of warning above 5 percent and failing above 10
percent mean-latency regression:

```sh
zig build benchmark-compare -- \
  --baseline benchmarks/results/baseline.jsonl \
  --candidate benchmarks/results/latest.jsonl \
  --runner zig \
  --json-output benchmarks/results/comparison.json \
  --report-output benchmarks/results/comparison.txt
```

## Current Coverage

- `primitive`
  - deterministic contiguous add
  - deterministic square matmul
  - deterministic nested-broadcast matmul that exercises the host fallback path
- `blas`
  - deterministic vector dot product
  - deterministic matrix-vector multiply
  - deterministic conv2d im2col lowering via batched matmul
- `autograd`
  - deterministic dot forward+backward
  - deterministic matvec forward+backward
- `memory`
  - host tensor cache allocation/free cycle high-water mark
  - synthetic MNIST training-step cache and graph-arena high-water mark
- `model-train`
  - synthetic MNIST-style MLP training step
  - synthetic CartPole-shaped DQN training step
  - synthetic two-layer GCN training step on a deterministic graph
- `model-infer`
  - synthetic MNIST-style MLP inference step
  - synthetic CartPole-shaped DQN inference step
  - synthetic two-layer GCN inference step on a deterministic graph

These model benchmarks mirror the repository's reference families while using
deterministic synthetic inputs, transitions, labels, and graphs so the suite
runs from a clean checkout without dataset downloads or simulator setup.

## Spec Format

Benchmark specs live under [`benchmarks/specs/`](./specs/) as JSON files.

Common fields:

- `id`: stable benchmark identifier
- `suite`: `primitive`, `blas`, `autograd`, `memory`, `model-train`, or `model-infer`
- `kind`: workload selector
- `dtype`: currently `f32`
- `warmup_iterations`
- `measured_iterations`
- `thread_count`
- `seed`
- `notes`

Workload-specific fields:

- `lhs_shape`, `rhs_shape` for primitive add/matmul, BLAS dot/matvec, and
  autograd dot/matvec backward
- `lhs_shape`, `rhs_shape`, `stride`, `padding`, and `dilation` for BLAS
  conv2d im2col lowering workloads
- `lhs_shape` plus `batch_size` for memory tensor-cache cycle workloads
- `batch_size`, `input_shape`, `label_shape` for batched model workloads
- `input_shape`, optional `label_shape`, and derived synthetic graph topology
  for GCN workloads
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
  - host provider (`accelerate`, `openblas`, or `mkl`)
  - configured thread count
  - optional host BLAS telemetry for Zig runs:
    - `dot_calls`
    - `matvec_calls`
    - `matmul_calls`
    - `bmm_acc_calls`
    - `direct_bmm_dispatches`
    - `fallback_bmm_dispatches`
    - `fallback_bmm_batches`
- setup latency
- measured latency summary:
  - min
  - median
  - mean
  - p95
  - max
  - throughput when applicable
- optional memory summary:
  - peak live bytes
  - final live bytes
  - peak graph arena bytes
  - final graph arena bytes
  - peak scratch bytes

## PyTorch Baseline

The optional PyTorch runner lives under
[`benchmarks/runners/pytorch/`](./runners/pytorch/).

If `torch` is not installed, the runner emits a `skipped` record instead of
failing the whole benchmark run. This keeps the default path dependency-light
while still allowing direct framework comparisons on prepared machines for the
BLAS, autograd, and model suites. Memory specs are Zig-only today because they
depend on Zigrad allocator and graph telemetry.

## Authoring

Benchmark authoring rules and validation expectations live in
[`benchmarks/AUTHORING.md`](./AUTHORING.md).
