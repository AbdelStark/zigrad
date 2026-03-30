# RFC-0007: Static Graph Optimization

Status: `Ready`
Priority: `P1`
Depends on: RFC-0006
Blocks: RFC-0004, RFC-0008, RFC-0009, RFC-0010, RFC-0011
Last updated: `2026-03-30`

## Summary

This RFC defines a stable graph IR and pass pipeline for optimizing captured
graphs before execution. It is the first compiler-adjacent layer in the roadmap
and is intended to provide correctness-preserving transformations, not dynamic
specialization.

## Motivation

Once lazy tensors can capture computations, the project needs a clear
intermediate representation and optimizer boundary. Static graph optimization is
the foundation for:

- better eager-derived execution,
- clean ONNX export/import mappings,
- inference translation,
- later dynamic specialization and compilation.

Without a stable optimization layer, higher-level compiler work will be forced
to target ad hoc runtime structures.

## Goals

- Define a stable, verifiable graph IR.
- Implement correctness-preserving optimization passes.
- Expose pass results and debug dumps.
- Support backend-aware but backend-independent optimization decisions.
- Measure optimization cost and payoff through the benchmark program.

## Non-Goals

- JIT specialization based on runtime guards.
- Introducing MLIR directly in this RFC.
- Rewriting all execution around the optimizer immediately.
- Performing unsafe, numerically changing transformations by default.

## IR Requirements

The graph IR must represent:

- typed tensor values,
- shapes and symbolic dimensions where available,
- devices and memory domains,
- constants and parameters,
- operation attributes,
- alias/view relationships,
- side-effect boundaries,
- source/debug names.

The IR must be serializable or at least dumpable for testing and diagnostics.

## Pass Categories

The initial pass set should include:

- dead code elimination,
- constant folding,
- algebraic simplification,
- canonicalization,
- common subexpression elimination where safe,
- transpose/layout simplification,
- fusion-candidate marking,
- memory planning annotations.

Each pass must declare:

- prerequisites,
- preserved invariants,
- invalidation behavior,
- debug output hooks.

## Verifier

The optimizer must ship with an IR verifier that checks:

- type consistency,
- shape consistency,
- single-definition rules,
- use-def integrity,
- alias/view invariants,
- backend compatibility constraints.

No optimization pass should run without verifier coverage before and after in
debug configurations.

## Pass Manager

Introduce a pass manager that supports:

- deterministic pass ordering,
- named pass pipelines,
- per-pass timing and stats,
- pass enable/disable configuration,
- textual dump after each pass in debug workflows.

## Execution Integration

Optimized graphs must remain executable through a defined lower/execute path.
The optimizer should not be a dead-end artifact generator. The initial execution
story can target the existing backend abstraction while keeping room for later
compiled lowering.

## Work Breakdown

### Workstream A: IR Definition

- formal node/value model,
- metadata carriers,
- verifier,
- graph dump format.

### Workstream B: Pass Infrastructure

- pass manager,
- pass registry,
- pass timing and statistics.

### Workstream C: Initial Optimizations

- DCE,
- constant folding,
- canonicalization,
- layout cleanup.

### Workstream D: Execution and Benchmarking

- lower optimized graphs to backend execution,
- compare optimized vs unoptimized performance and correctness.

## Testing Plan

- verifier unit tests,
- pass-specific golden tests,
- end-to-end parity tests on captured graphs,
- benchmark coverage for pass overhead and performance gains,
- stress tests on large graphs from example models.

## Acceptance Criteria

- Captured graphs can be represented in the IR and verified.
- At least three optimization passes are implemented and benchmarked.
- Optimized execution is numerically equivalent on supported workloads.
- The IR and pass manager are reused by at least one later RFC.

## Risks

- Alias and view semantics can complicate seemingly simple optimizations.
- Overly abstract IR design can stall implementation.
- Optimization overhead may exceed benefit on small graphs if pipelines are not
  configurable.

## Open Questions

- How much symbolic shape reasoning is required in the first version?
- Should memory planning be annotations in this RFC or a later dedicated pass?
- Do we permit backend-specific passes in the base pass manager?

## Agentic Context

### 2026-03-30 Graph IR Foundation, Verifier, Pass Manager, and DCE

- Completed:
  - Defined a stable graph IR in
    [`src/graph_ir.zig`](../../src/graph_ir.zig)
    with typed `Value` nodes (id, dtype, shape, device, storage, label,
    requires_grad, defining_op), `Op` nodes (id, name, operands, results,
    attributes), and graph-level input/output tracking.
  - Added `DType` enum with `fromName()` for lowering lazy-session dtype
    strings into typed IR values.
  - Implemented `GraphIR.fromSession()` that lowers `lazy.Session` captures
    into the IR, mapping source/external_input records to graph inputs and
    computed tensor records to op+value pairs. Materialized tensors become
    graph outputs.
  - Implemented an IR verifier that checks: no duplicate value/op IDs, all
    operand references resolve, all defining ops exist, and the dataflow
    graph is acyclic (DFS with cycle detection).
  - Implemented a `PassManager` with deterministic ordering, per-pass timing
    statistics, and optional pre/post verification after each pass.
  - Implemented DCE (dead code elimination) as the first real optimization
    pass — removes ops whose results are not reachable from graph outputs,
    and removes orphaned values.
  - Added structural stubs for constant folding and algebraic simplification
    passes, ready for host-side evaluation wiring.
  - Added a `writeSummary()` text dump for the IR in SSA-like notation with
    typed values, op names, and attributes.
  - Exported `graph_ir` through
    [`src/zigrad.zig`](../../src/zigrad.zig),
    and added public attribute helper wrappers in
    [`src/lazy.zig`](../../src/lazy.zig).
  - Added five regression tests: session-to-IR lowering, dangling operand
    verifier detection, DCE removing dead ops, pass manager execution with
    stats, and summary output format.
- Remains:
  - Constant folding with host-side evaluation (requires execution bridge).
  - Algebraic simplification with constant value tracking.
  - Additional passes: CSE, transpose/layout simplification, fusion marking.
  - Execution bridge: lower optimized IR back to device dispatch.
  - Benchmark coverage comparing optimized vs unoptimized execution.
- Blockers:
  - Constant folding and algebraic simplification require either embedding
    constant data in the IR or an execution bridge that can evaluate ops at
    optimization time.
- Validation performed:
  - `zig fmt src/graph_ir.zig src/lazy.zig src/zigrad.zig`
  - `zig build test` — all tests pass including five new graph IR tests
  - `zig build test-example-smoke` — no regressions

