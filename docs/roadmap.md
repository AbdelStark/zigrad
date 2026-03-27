# Zigrad Roadmap and RFC Index

This document decomposes the public roadmap into implementation-ready RFCs and
specifications. It should be treated as the canonical planning index for major
platform, compiler, interoperability, and example work.

The older [roadmap.norg](./roadmap.norg)
remains useful historical context. The files listed here are the new execution
documents we will implement against.

## Status Legend

- `Draft`: scoped, but further validation is required before coding.
- `Ready`: sufficiently specified to begin implementation.
- `Planned`: accepted into the roadmap, but intentionally sequenced behind
  dependencies.
- `Exploratory`: worth pursuing, but requires spikes or external validation.
- `Blocked`: accepted, but cannot start until predecessor RFCs land.

## Program Principles

- Preserve Zigrad's research-first, eager-by-default workflow.
- Add complexity only where it unlocks measurable capability or performance.
- Prefer staged delivery over large rewrites hidden behind a single branch.
- Make every performance claim reproducible through the benchmark program.
- Keep optional integrations optional at build time and explicit at runtime.
- Treat interoperability as product surface, not just import/export plumbing.

## Workstreams

### Phase 0: Measurement and Backend Foundations

- [RFC-0001](./rfcs/0001-benchmarking-program.md)
  defines the benchmark harness, regression policy, and result schema.
- [RFC-0002](./rfcs/0002-onemkl-host-backend.md)
  expands the CPU backend story around oneMKL and provider selection.
- [RFC-0003](./rfcs/0003-cuda-backend.md)
  brings CUDA from experimental to supported.

### Phase 1: Interop and Execution Model

- [RFC-0004](./rfcs/0004-onnx-interop.md)
  covers ONNX import/export.
- [RFC-0005](./rfcs/0005-ggml-gguf-interop.md)
  covers ggml/GGUF model and weight interoperability.
- [RFC-0006](./rfcs/0006-lazy-tensors.md)
  introduces deferred execution as an opt-in execution mode.
- [RFC-0012](./rfcs/0012-examples-and-reference-models.md)
  defines the reference example portfolio that will validate the stack.

### Phase 2: Optimization and Compilation

- [RFC-0007](./rfcs/0007-static-graph-optimization.md)
  defines a verifiable optimization pipeline for captured graphs.
- [RFC-0008](./rfcs/0008-dynamic-graph-compiler.md)
  adds specialization and compilation for dynamic graphs.
- [RFC-0009](./rfcs/0009-mlir-lowering-pipeline.md)
  introduces MLIR as an optional lowering and interchange layer.

### Phase 3: Inference Translation and External Compilers

- [RFC-0010](./rfcs/0010-zml-inference-bridge.md)
  defines inference-only translation into ZML.
- [RFC-0011](./rfcs/0011-apache-tvm-integration.md)
  defines optional TVM integration for ahead-of-time and autotuned execution.

## RFC Matrix

| ID | Title | Status | Priority | Depends on | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC-0001 | Standardized Benchmarking Program | `Ready` | P0 | None | Harness, JSONL output, comparison/regression tooling, authoring guide, smoke CI, and synthetic BLAS/autograd/memory/MNIST/DQN/GCN coverage are landed; future CUDA/compiler/interop suites remain. |
| RFC-0002 | oneMKL Host Backend | `Ready` | P0 | RFC-0001 | Expands host performance beyond the current baseline. |
| RFC-0003 | CUDA Backend | `Ready` | P0 | RFC-0001 | Turns experimental CUDA into a supported execution backend. |
| RFC-0004 | ONNX Interop | `Planned` | P1 | RFC-0001, RFC-0007 | Best treated as import/export on top of a stable graph IR. |
| RFC-0005 | ggml/GGUF Interop | `Planned` | P1 | RFC-0001, RFC-0012 | Critical for LLM examples and inference compatibility. |
| RFC-0006 | Lazy Tensors | `Planned` | P1 | RFC-0001, RFC-0002, RFC-0003 | Introduces capture without breaking eager execution. |
| RFC-0007 | Static Graph Optimization | `Planned` | P1 | RFC-0006 | First optimization layer and foundation for compiler work. |
| RFC-0008 | Dynamic Graph Compiler | `Draft` | P2 | RFC-0006, RFC-0007 | Specialization and caching for dynamic workloads. |
| RFC-0009 | MLIR Lowering Pipeline | `Exploratory` | P2 | RFC-0007, RFC-0008 | Optional compiler interoperability layer. |
| RFC-0010 | ZML Inference Bridge | `Draft` | P2 | RFC-0007 | Enables inference handoff to ZML for pure serving flows. |
| RFC-0011 | Apache TVM Integration | `Exploratory` | P3 | RFC-0001, RFC-0007, RFC-0009 | External compiler/autotuning path. |
| RFC-0012 | Examples and Reference Models | `Planned` | P1 | RFC-0001, RFC-0002, RFC-0003 | Program-level validation for all major capabilities. |

## Recommended Implementation Order

1. RFC-0001 Standardized Benchmarking Program
2. RFC-0002 oneMKL Host Backend
3. RFC-0003 CUDA Backend
4. RFC-0012 Examples and Reference Models, initial CV/RL refresh
5. RFC-0006 Lazy Tensors
6. RFC-0007 Static Graph Optimization
7. RFC-0004 ONNX Interop
8. RFC-0005 ggml/GGUF Interop
9. RFC-0010 ZML Inference Bridge
10. RFC-0008 Dynamic Graph Compiler
11. RFC-0009 MLIR Lowering Pipeline
12. RFC-0011 Apache TVM Integration

## Definition of Done for the Roadmap Program

The roadmap is considered complete only when:

- All RFCs have been implemented or explicitly closed with replacement plans.
- Every accepted feature has correctness tests and benchmark coverage.
- CPU and CUDA execution paths are both validated on representative models.
- Import/export and translation paths have round-trip or conformance coverage.
- At least one LLM, one physics/control example, and one RL example are kept
  green in CI smoke form.
- The benchmark program can reproduce published performance claims from a clean
  checkout.

## RFC Authoring Conventions

Every RFC in this folder set must maintain:

- an explicit status,
- concrete dependency and blocking relationships,
- a phased delivery plan,
- measurable acceptance criteria,
- test and benchmark requirements,
- a section describing what will not be done in the RFC.

## Agentic Context

### RFC-0001 Snapshot

- Completed:
  - Added the benchmark harness under [`benchmarks/`](../benchmarks/) with JSON
    spec loading, JSON-lines result emission, runtime/system/backend metadata
    capture, and build entrypoints.
  - Landed initial `primitive`, `model-train`, and `model-infer` specs covering
    deterministic add, deterministic matmul, synthetic MNIST-style MLP train,
    synthetic MNIST-style MLP infer, synthetic CartPole-style DQN train/infer,
    and synthetic two-layer GCN train/infer.
  - Added an optional PyTorch baseline runner that emits `skipped` records when
    `torch` is unavailable and now understands the expanded model benchmark
    kinds.
  - Added benchmark smoke CI in
    [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml).
  - Added benchmark result comparison tooling with threshold-based regression
    classification, `zig build benchmark-compare`, and JSON/text comparison
    outputs for local runs and CI.
  - Added the benchmark authoring guide in
    [`benchmarks/AUTHORING.md`](../benchmarks/AUTHORING.md).
  - Updated smoke CI to benchmark the base revision and current checkout on the
    same runner, compare results, and upload comparison artifacts.
- Remains:
  - Cross-platform baseline data with `torch` installed and broader
    CUDA/compiler/interop workload coverage.
  - Published benchmark reporting pages or artifacts beyond uploaded CI JSONL.
- Blockers:
  - No local PyTorch installation was present in this run, so baseline execution
    validated only the explicit `skipped` path.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark-models`
  - `zig build benchmark-compare -- --help`
  - `zig build benchmark`
  - `zig build benchmark-models -- --baseline pytorch`
  - `zig build benchmark-compare -- --baseline benchmarks/results/latest.jsonl --candidate benchmarks/results/latest.jsonl --runner zig --json-output benchmarks/results/comparison.json --report-output benchmarks/results/comparison.txt`
  - `zig build benchmark-primitive`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`
  - `python3 -c "import json, pathlib; print(pathlib.Path('benchmarks/results/comparison.json').exists())"`

### RFC-0001 2026-03-27 BLAS + Autograd Coverage

- Completed:
  - Added dedicated `blas` and `autograd` benchmark suites with deterministic
    dot and matvec workloads under [`benchmarks/specs/`](../benchmarks/specs/).
  - Extended the harness manifest, CLI grouping, and `zig build` entrypoints
    so `benchmark`, `benchmark-blas`, and `benchmark-autograd` execute the new
    coverage.
  - Added Zig workload implementations for BLAS forward coverage and autograd
    backward coverage in
    [`benchmarks/src/workload.zig`](../benchmarks/src/workload.zig).
  - Expanded the optional PyTorch baseline runner to emit comparable records
    for the new BLAS and autograd benchmark kinds.
- Remains:
  - Add CUDA-targeted suites and backend-specific parity checks once RFC-0003
    work begins.
  - Add compiler, interop, and memory suites to cover the remaining RFC-0001
    benchmark taxonomy.
- Blockers:
  - No local `torch` install was available during this run, so PyTorch parity
    was validated through runner compilation and skip-path behavior rather than
    executed framework comparisons.
- Validation:
  - `zig build test`
  - `zig build benchmark-blas`
  - `zig build benchmark-autograd`
  - `zig build benchmark`
  - `zig build benchmark -- --baseline pytorch --group blas`
  - `zig build benchmark -- --baseline pytorch --group autograd`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`

### RFC-0001 2026-03-27 Memory Coverage

- Completed:
  - Added host caching allocator telemetry and graph arena capacity reporting
    so benchmark records can include memory high-water marks.
  - Added a dedicated `memory` suite with a tensor cache cycle workload and a
    synthetic MNIST training-step workload.
  - Extended benchmark comparison output to include memory deltas and fail on
    threshold-exceeding memory regressions.
- Remains:
  - Extend the same benchmark taxonomy to CUDA memory accounting and future
    compiler/interop workloads.
- Blockers:
  - PyTorch baseline coverage still does not apply to Zigrad-native memory
    telemetry workloads.
- Validation:
  - `zig build test`
  - `zig build benchmark-memory`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/memory.jsonl --candidate benchmarks/results/memory.jsonl --runner zig --json-output benchmarks/results/memory-comparison.json --report-output benchmarks/results/memory-comparison.txt`
