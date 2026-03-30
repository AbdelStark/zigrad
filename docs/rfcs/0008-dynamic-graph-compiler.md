# RFC-0008: Dynamic Graph Compiler

Status: `Draft`
Priority: `P2`
Depends on: RFC-0006, RFC-0007
Blocks: RFC-0009
Last updated: `2026-03-30`

## Summary

This RFC defines a dynamic graph compiler that specializes captured workloads
based on runtime-observed properties such as shapes, dtypes, device placement,
and execution mode. It adds caching, guard evaluation, compiled segment reuse,
and a fallback path to optimized or eager execution.

## Motivation

Zigrad's value proposition includes dynamic, research-friendly workflows. A
dynamic graph compiler should improve performance without requiring users to
rewrite models into a fully static style. This is especially relevant for RL,
control, sequence models, and workloads with moderate shape variability.

## Goals

- Compile reusable graph segments from dynamic workloads.
- Guard compiled artifacts with explicit specialization predicates.
- Cache compiled segments and reuse them safely.
- Preserve a fallback path to optimized static or eager execution.
- Expose compile cost and cache hit rates through benchmark instrumentation.

## Non-Goals

- Whole-program compilation of arbitrary Zig user code.
- Eliminating eager execution.
- Cross-process distributed compilation cache in the first milestone.
- MLIR or TVM integration in this RFC.

## Compilation Model

The compiler should:

1. capture a candidate subgraph,
2. normalize it through static optimization passes,
3. derive specialization keys,
4. compile executable segments,
5. install runtime guards,
6. dispatch to cached compiled code or fall back when guards fail.

## Specialization Dimensions

Likely specialization keys include:

- shape and rank,
- dtype,
- device/backend,
- layout and contiguity,
- constant attributes and selected configuration flags.

The system must explicitly state which dimensions participate in caching.

## Runtime Guards

Guards must be:

- cheap to evaluate,
- deterministic,
- serializable or inspectable for debugging,
- tied to a clear fallback path.

Guard failures should increment counters and optionally emit diagnostics in debug
mode so users understand why compilation did not apply.

## Cache Design

The initial cache can be in-memory and process-local. It must support:

- lookup by specialization key,
- lifetime management,
- invalidation on backend/config changes,
- statistics for hit/miss/compile counts.

Persistent caches are a future enhancement, not a requirement for the first
delivery.

## Execution Surface

Compiled segments must interoperate with:

- eager tensors,
- lazy tensors,
- autograd bookkeeping,
- host and CUDA backends where supported.

The compiler may start with forward-only segment compilation before taking on
backward graphs.

## Work Breakdown

### Workstream A: Candidate Extraction

- define what graph fragments are compilable,
- identify capture boundaries,
- preserve source and debug metadata.

### Workstream B: Specialization and Guards

- key derivation,
- guard evaluation,
- cache lookup.

### Workstream C: Code Generation Path

- compiled executable representation,
- lowering from optimized IR,
- runtime invocation boundary.

### Workstream D: Autograd and Debugging

- forward/backward integration strategy,
- cache statistics,
- debug dumps and fallback traces.

## Testing Plan

- guard correctness tests,
- cache behavior tests,
- eager-vs-compiled parity tests,
- benchmark coverage for compile cost, cache hit rate, and steady-state speedup,
- example integration on at least one RL or sequence-style workload.

## Acceptance Criteria

- The system compiles and reuses at least one non-trivial graph segment.
- Guard failures fall back safely and transparently.
- Benchmarks report compile latency and steady-state performance.
- At least one workload shows a measurable steady-state speedup over optimized
  but uncompiled execution.

## Risks

- Compile overhead may dominate on short-lived workloads.
- Incorrect guards can lead to subtle wrong results.
- Backward-graph compilation may be significantly harder than forward segments.

## Open Questions

- Should compilation initially target only inference or both inference and
  training?
- How small can a compiled segment be before the cost is not worth it?
- Is the first backend target handwritten codegen, MLIR, or another IR?

## Agentic Context

### 2026-03-30 Dependency Status Update

- Dependencies now available:
  - **RFC-0006 (Lazy Tensors):** Deferred forward execution via thunk queue
    is landed. Operations can be captured and replayed. The lazy session
    records op names, dtypes, shapes, devices, parent edges, and structured
    attributes. See `src/lazy.zig`.
  - **RFC-0007 (Static Graph Optimization):** Graph IR with typed Value/Op
    nodes in SSA form, verifier (use-def + acyclicity), pass manager with
    timing, and a working DCE pass are all landed. See `src/graph_ir.zig`.
- What this RFC can now build on:
  - The `GraphIR.fromSession()` lowering provides a stable IR from lazy
    captures. The pass manager provides the infrastructure for adding
    compilation passes. The execution bridge (RFC-0007 M-1, not yet landed)
    will provide the ability to execute optimized/compiled graphs.
- Blockers:
  - This RFC remains `Draft` because it needs a scoping spike to decide
    the compilation target (handwritten codegen vs MLIR) and minimum viable
    segment size before implementation begins.
  - The execution bridge (RFC-0007 M-1) should land before this RFC starts,
    as compiled segments need an execution path.
- Recommended next step:
  - A scoping spike that captures one of the reference models (e.g.,
    char-LM or MNIST) in deferred mode, lowers to IR, and measures the
    overhead of capture + IR construction to establish a baseline for what
    compilation needs to improve on.
