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
| RFC-0001 | Standardized Benchmarking Program | `Ready` | P0 | None | Harness, JSONL output, initial primitive/MNIST specs, and smoke CI are landed; comparison/regression policy remains. |
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
    and synthetic MNIST-style MLP infer.
  - Added an optional PyTorch baseline runner that emits `skipped` records when
    `torch` is unavailable.
  - Added benchmark smoke CI in
    [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml).
- Remains:
  - Result-to-result comparison tooling and regression thresholds.
  - Broader model coverage including DQN/GCN and cross-platform baseline data.
  - Published benchmark reporting pages or artifacts beyond uploaded CI JSONL.
- Blockers:
  - No local PyTorch installation was present in this run, so baseline execution
    validated only the explicit `skipped` path.
- Validation:
  - `zig build test`
  - `zig build benchmark-primitive`
  - `zig build benchmark-models`
  - `zig build benchmark`
