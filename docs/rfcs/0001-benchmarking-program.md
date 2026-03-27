# RFC-0001: Standardized Benchmarking Program

Status: `Ready`  
Priority: `P0`  
Depends on: None  
Blocks: RFC-0002, RFC-0003, RFC-0004, RFC-0005, RFC-0006, RFC-0012  
Last updated: `2026-03-27`

## Summary

Zigrad needs a first-class benchmarking program that measures kernels, model
training, model inference, compiler passes, memory behavior, and cross-framework
comparisons in a reproducible way. This RFC defines the benchmark harness,
result schema, comparison policy, CI strategy, and publication process that all
future optimization work will use.

## Motivation

The roadmap contains multiple performance-sensitive initiatives: oneMKL, CUDA,
lazy execution, graph optimization, dynamic compilation, MLIR lowering, ZML
translation, and TVM integration. Without a consistent benchmark program, it
will be impossible to determine whether these efforts actually improve latency,
throughput, memory use, or developer ergonomics.

The current project has ad hoc performance artifacts in `docs/`, but it does
not yet define:

- a stable suite layout,
- required hardware metadata,
- baseline frameworks,
- noise controls,
- regression thresholds, or
- a result format suitable for automation.

## Goals

- Provide a single benchmark entrypoint and directory layout.
- Cover microbenchmarks, model benchmarks, and end-to-end workflows.
- Store benchmark results in a machine-readable format.
- Make framework comparisons reproducible across machines and commits.
- Support both local exploration and CI-driven regression detection.
- Record hardware, backend, dtype, dataset, and workload metadata alongside
  every result.

## Non-Goals

- Building a benchmark web service in the first iteration.
- Replacing existing docs images immediately.
- Enforcing exact cross-machine performance comparability.
- Running the full benchmark suite on every PR.

## Benchmark Taxonomy

The program will define the following suites:

- `primitive`: scalar ops, elementwise ops, reductions, broadcasting, reshape,
  transpose, gather, indexing, and memory copies.
- `blas`: GEMM, batched GEMM, matvec, conv lowering where relevant.
- `autograd`: forward+backward for representative graphs and module stacks.
- `model-train`: MNIST MLP/CNN, DQN, GCN, and future reference models.
- `model-infer`: eager inference, lazy inference, translated inference.
- `compiler`: graph capture, optimization pass cost, compile latency, cache hit
  behavior.
- `interop`: ONNX import cost, GGUF load cost, ZML translation cost.
- `memory`: peak bytes, allocator churn, host-device transfer volume.

## Proposed Layout

```text
benchmarks/
  README.md
  specs/
    primitive/
    model-train/
    model-infer/
  runners/
    zig/
    pytorch/
    zml/
  datasets/
    manifests/
  results/
    .gitignore
  scripts/
```

`zig build benchmark` will dispatch into benchmark suites. Additional steps such
as `zig build benchmark-primitive` and `zig build benchmark-models` may be added
for convenience.

## Result Schema

Every result record must contain:

- benchmark id,
- git commit SHA,
- dirty tree flag,
- Zig version,
- benchmark harness version,
- operating system and kernel version,
- CPU model, core count, frequency policy,
- GPU model, driver version, compute capability if applicable,
- backend configuration: Accelerate/OpenBLAS/oneMKL/CUDA/etc,
- dtype,
- tensor shapes,
- batch size,
- warmup count,
- measured iterations,
- min/median/mean/p95/max latency,
- throughput where relevant,
- peak memory where relevant,
- notes for skipped or unsupported cases.

The initial on-disk format will be JSON lines. CSV export can be layered on top.

## Baseline Framework Policy

The benchmarking program will compare Zigrad against:

- PyTorch for CPU and CUDA where feature parity exists,
- ZML for inference-oriented workflows once the bridge exists,
- optionally TensorFlow or JAX only when a benchmark specifically requires it.

Baseline runners must record:

- framework version,
- exact model definition,
- backend flags,
- dtype,
- optimizer settings,
- dataset preprocessing choices.

## Reproducibility Rules

- Random seeds must be fixed and recorded.
- Benchmark specs must declare data source and preprocessing.
- CPU benchmarks must support thread count pinning.
- CUDA benchmarks must separate compile/warmup from measurement.
- Benchmarks that allocate model weights must distinguish one-time setup cost
  from steady-state iteration cost.
- Results used in docs must be reproducible from the published spec.

## CI Strategy

PR CI will run a smoke subset:

- one primitive suite,
- one CPU model training suite,
- one inference suite,
- one correctness-checked performance sanity benchmark.

Scheduled CI will run the broader suite on dedicated runners where available.
Regression gating should use broad thresholds first, for example:

- fail on greater than 10 percent regression on stable smoke benchmarks,
- warn on greater than 5 percent regression,
- never gate on noisy exploratory suites.

## Deliverables

- Benchmark directory structure and harness entrypoints.
- JSON result schema and readers.
- Hardware and runtime metadata collection utilities.
- Benchmark spec files for existing MNIST, DQN, and tensor primitives.
- Baseline PyTorch runners for the first comparable workloads.
- CI smoke benchmark integration.
- Documentation for adding a new benchmark.

## Milestones

### Milestone A: Harness

- [x] Create benchmark directory structure.
- [x] Define benchmark manifest schema.
- [x] Add local CLI/build entrypoints.

### Milestone B: Initial Coverage

- [x] Primitive suite for representative tensor ops.
- [x] MNIST train and inference benchmarks using deterministic synthetic data.
- [ ] DQN benchmark skeleton.
- [x] Optional PyTorch baseline runner hook for model benchmarks.

### Milestone C: Regression Policy

- [x] JSON result comparison utility.
- [x] CI smoke suite.
- [x] Published benchmark authoring guide.

## Acceptance Criteria

- `zig build benchmark` runs from a clean checkout.
- At least three benchmark suites emit schema-valid JSON.
- Results include hardware and backend metadata.
- One baseline PyTorch comparison exists for a model benchmark.
- CI runs a smoke benchmark set and reports regression deltas.

## Risks

- Benchmarks can become noisy if thread placement and thermal state are not
  controlled.
- Framework comparisons can become unfair if preprocessing diverges.
- CI hardware variability can make strict thresholds brittle.

## Open Questions

- Should benchmark specs be JSON, Zig source, or both?
- Do we want per-operator flop accounting in the initial version?
- Should result archives live in the main repo or an external artifact store?

## Agentic Context

### 2026-03-27

- Completed:
  - Added the benchmark harness under [`benchmarks/`](../benchmarks/) with
    manifest parsing, deterministic workload execution, JSON-lines result
    emission, metadata capture, and `zig build benchmark*` entrypoints.
  - Added initial benchmark specs for primitive add/matmul and synthetic
    MNIST-style MLP train/infer workloads.
  - Added an optional PyTorch baseline runner that emits compatible `skipped`
    records when `torch` is unavailable.
  - Added CI smoke execution in
    [`.github/workflows/benchmark-smoke.yml`](../../.github/workflows/benchmark-smoke.yml).
  - Added the benchmark comparison utility under
    [`benchmarks/src/compare.zig`](../../benchmarks/src/compare.zig) with
    threshold-based pass/warn/fail classification, JSON and text reports, and a
    `zig build benchmark-compare` entrypoint.
  - Added the benchmark authoring guide in
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md).
  - Updated benchmark smoke CI to compare the current checkout against the base
    revision on the same runner and upload comparison artifacts.
- Remains:
  - Expand benchmark coverage to DQN, GCN, and future CUDA/compiler/interop
    suites.
  - Add richer reporting and historical storage beyond per-run JSON/text
    artifacts.
- Blockers:
  - No local `torch` install was available during this run, so the PyTorch
    runner was only validated through its explicit skip path.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-compare -- --help`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/latest.jsonl --candidate benchmarks/results/latest.jsonl --runner zig --json-output benchmarks/results/comparison.json --report-output benchmarks/results/comparison.txt`
  - `zig build benchmark-primitive`
  - `zig build benchmark-models`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`
  - `python3 -c "import json, pathlib; print(pathlib.Path('benchmarks/results/comparison.json').exists())"`
- Exact commands:
  - `zig build test`
  - `zig build benchmark-compare -- --help`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/latest.jsonl --candidate benchmarks/results/latest.jsonl --runner zig --json-output benchmarks/results/comparison.json --report-output benchmarks/results/comparison.txt`
  - `zig build benchmark-primitive`
  - `zig build benchmark-models`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`
  - `python3 -c "import json, pathlib; print(pathlib.Path('benchmarks/results/comparison.json').exists())"`
