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
zig build benchmark-validate
zig build test-benchmark-smoke
zig build test-benchmark-cuda-request-smoke
zig build test-benchmark-baseline-smoke
zig build test-benchmark-publication-smoke
zig build benchmark-publication-bundle -- --candidate-jsonl benchmarks/results/latest.jsonl --summary-output benchmarks/results/publication-summary.md --manifest-output benchmarks/results/publication-manifest.json
zig build benchmark-compare -- --baseline benchmarks/results/baseline.jsonl --candidate benchmarks/results/latest.jsonl
zig build benchmark-provider-report -- --input benchmarks/results/accelerate.jsonl --input benchmarks/results/openblas.jsonl --baseline-provider accelerate
zig build benchmark-thread-report -- --input benchmarks/results/thread-sweep.jsonl --baseline-thread-count 1
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
zig build benchmark -- --spec benchmarks/specs/model-infer/char-lm-synthetic.json
zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json
zig build benchmark -- --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json --thread-count 1 --thread-count 2 --thread-count 4
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

## Contract Validation

Use the benchmark validator when you need to prove that committed specs and/or
emitted JSONL artifacts still satisfy the RFC-0001 contract:

```sh
zig build benchmark-validate
zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output .zig-cache/zigrad-benchmark-validate.jsonl
zig build benchmark-validate -- --input .zig-cache/zigrad-benchmark-validate.jsonl
zig build test-benchmark-smoke
zig build test-benchmark-cuda-request-smoke
zig build test-benchmark-baseline-smoke
zig build test-benchmark-publication-smoke
```

`benchmark-validate` checks the selected spec tree when no `--input` is
provided. When `--input` paths are supplied, it validates each JSONL record
against the referenced checked-in spec, checks summary-stat invariants, and
rejects duplicate result identities within a file. `test-benchmark-smoke`
drives one representative checked-in spec per suite through the real benchmark
harness and then runs the validator on the generated artifact.
`test-benchmark-cuda-request-smoke` exercises checked-in CUDA-targeted specs
through the real harness and requires non-CUDA hosts to emit explicit
schema-valid `skipped` records instead of aborting.
`test-benchmark-baseline-smoke` covers the external baseline-runner contract by
requiring successful baseline emission to stay schema-valid while malformed or
missing runners degrade into explicit `failed` records instead of disappearing.
`test-benchmark-publication-smoke` builds on that by generating smoke-scale
comparison, provider-report, thread-report, and publication-bundle artifacts
and failing if those publication outputs are missing, empty, or structurally
invalid.

For RFC-0002 host BLAS work, collect one JSONL file per provider and then
generate a provider matrix report:

```sh
zig build benchmark -Dhost_blas=accelerate -- --output benchmarks/results/accelerate.jsonl
zig build benchmark -Dhost_blas=openblas -- --output benchmarks/results/openblas.jsonl
zig build benchmark -Dhost_blas=mkl -Dmkl_include_dir=/opt/intel/oneapi/mkl/latest/include -Dmkl_library_dir=/opt/intel/oneapi/mkl/latest/lib -- --output benchmarks/results/mkl.jsonl
zig build benchmark-provider-report -- \
  --input benchmarks/results/accelerate.jsonl \
  --input benchmarks/results/openblas.jsonl \
  --input benchmarks/results/mkl.jsonl \
  --runner zig \
  --baseline-provider accelerate \
  --markdown-output benchmarks/results/host-provider-report.md \
  --json-output benchmarks/results/host-provider-report.json
```

The provider report only considers host-device records for the selected runner,
groups them by benchmark id and configured thread count, and then emits
Markdown/JSON tables with raw metrics plus delta-vs-baseline speedups.

## Thread Scaling Reports

RFC-0002 thread-scaling validation now has a dedicated workflow. Run one or
more specs across repeated `--thread-count` overrides, then summarize the
results with the scaling report:

```sh
zig build benchmark -- \
  --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json \
  --thread-count 1 \
  --thread-count 2 \
  --thread-count 4 \
  --output benchmarks/results/thread-sweep.jsonl
zig build benchmark-thread-report -- \
  --input benchmarks/results/thread-sweep.jsonl \
  --baseline-thread-count 1 \
  --markdown-output benchmarks/results/thread-scaling.md \
  --json-output benchmarks/results/thread-scaling.json
```

The scaling report groups host records by benchmark id and provider, orders the
rows by thread count, and computes latency deltas, speedups, and scaling
efficiency relative to the selected baseline thread count. If no
`--baseline-thread-count` is provided, the smallest available successful thread
count in each group becomes the baseline automatically.

## Publication Bundles

When you need a single publication artifact set for CI or docs workflows, point
the bundle tool at the emitted JSONL files and derived reports:

```sh
zig build benchmark-publication-bundle -- \
  --candidate-jsonl benchmarks/results/latest.jsonl \
  --baseline-jsonl benchmarks/results/baseline.jsonl \
  --extra-results-jsonl benchmarks/results/thread-sweep.jsonl \
  --comparison-json benchmarks/results/comparison.json \
  --comparison-text benchmarks/results/comparison.txt \
  --thread-report-json benchmarks/results/thread-scaling.json \
  --thread-report-markdown benchmarks/results/thread-scaling.md \
  --manifest-output benchmarks/results/publication-manifest.json \
  --summary-output benchmarks/results/publication-summary.md
```

The tool validates that comparison/provider/thread reports still reference the
supplied JSONL inputs, records artifact sizes and runtime fingerprints, and
emits both a machine-readable manifest and a Markdown summary for humans.

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
  - synthetic char-level causal language model training step
  - synthetic CartPole-shaped DQN training step
  - CUDA-targeted synthetic CartPole-shaped DQN training step spec
  - synthetic two-layer GCN training step on a deterministic graph
- `model-infer`
  - synthetic MNIST-style MLP inference step
  - synthetic char-level causal language model inference step
  - CUDA-targeted synthetic MNIST-style MLP inference step spec
  - synthetic CartPole-shaped DQN inference step
  - synthetic two-layer GCN inference step on a deterministic graph

These model benchmarks mirror the repository's reference families while using
deterministic synthetic inputs, transitions, labels, and graphs so the suite
runs from a clean checkout without dataset downloads or simulator setup. The
char-LM workloads mirror [`examples/char-lm/`](../examples/char-lm/) with
one-hot causal windows and next-token labels.

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
- `device` (optional, defaults to `host`; use `cuda` or `cuda:<index>` for
  CUDA-targeted specs)
- `seed`
- `provenance`:
  - `data_source`: explicit workload input origin such as `synthetic.splitmix64`
  - `preprocessing`: ordered preprocessing/materialization steps that shape the benchmark inputs
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

CUDA-targeted specs are part of the committed benchmark contract. On a build
without CUDA support, or on a host with no CUDA device, the Zig runner emits
an explicit `skipped` record for those specs instead of failing the whole
harness. Successful CUDA runs carry structured device metadata in
`backend.cuda`. The `memory` suite remains host-only for now, and PyTorch
baseline rows for CUDA-targeted specs currently emit explicit `skipped`
records.

The harness also accepts repeated `--thread-count <n>` CLI overrides. When
present, it duplicates each selected spec across the requested thread counts
without changing the checked-in JSON spec files.

## Output Schema

Each run emits one JSON object per benchmark with:

- benchmark id, checked-in spec path, suite, kind, runner, and status
- dtype, batch size, warmup/measured iteration counts, and seed
- shape metadata
- benchmark provenance:
  - declared data source
  - declared preprocessing steps
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
  - CPU frequency policy when discoverable
  - total memory when discoverable
- backend metadata:
  - device kind
  - host provider (`accelerate`, `openblas`, or `mkl`)
  - configured thread count
  - captured thread-environment hints when present:
    - `VECLIB_MAXIMUM_THREADS`
    - `OPENBLAS_NUM_THREADS`
    - `OMP_NUM_THREADS`
    - `MKL_NUM_THREADS`
    - `MKL_DYNAMIC`
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
depend on Zigrad allocator and graph telemetry. If the runner cannot be
launched, exits non-zero, or emits malformed JSONL, the harness now records a
structured `failed` baseline row for the spec so validator and comparison flows
can surface the problem explicitly.

## Authoring

Benchmark authoring rules and validation expectations live in
[`benchmarks/AUTHORING.md`](./AUTHORING.md).
