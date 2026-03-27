# RFC-0006: Lazy Tensors

Status: `Planned`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0002, RFC-0003  
Blocks: RFC-0007, RFC-0008, RFC-0010  
Last updated: `2026-03-27`

## Summary

This RFC introduces lazy tensors as an opt-in execution mode that records
operation intent, delays materialization, and creates a stable boundary for
graph optimization and compilation. Eager execution remains the default user
experience.

## Motivation

Several roadmap items require a notion of graph capture:

- static graph optimization,
- dynamic graph compilation,
- inference translation,
- compiler interop.

Attempting those features directly on purely eager operations leads to fragile
ad hoc tracing. Lazy tensors provide a principled mechanism for building graph
fragments while preserving the ergonomics of eager-first research workflows.

## Goals

- Add a lazy execution mode without breaking eager defaults.
- Capture tensor operations into a graph form suitable for optimization.
- Define explicit materialization boundaries and debug visibility.
- Preserve autograd semantics in both eager and lazy mode.
- Support CPU and CUDA backends under the same high-level model.

## Non-Goals

- Replacing eager execution.
- Implementing the full compiler in this RFC.
- Global graph capture of arbitrary host-side control flow.
- Automatic distributed scheduling.

## User Model

Users should be able to:

- create lazy tensors explicitly,
- enable lazy capture for a scoped region,
- force realization/materialization when needed,
- inspect the captured graph in debug workflows.

The default path should remain eager unless the user or higher-level system
opts into lazy mode.

## Core Concepts

- `LazyTensor`: a tensor handle backed by a graph node rather than immediate
  storage.
- `Thunk` or `PendingOp`: captured operation plus inputs, attributes, and
  inferred output metadata.
- `Materialization`: conversion of one or more lazy values into concrete backend
  buffers.
- `Graph Session`: an ownership boundary for captured lazy subgraphs.

## Design Requirements

- Operations must be representable independently of the current backend.
- Captured nodes must carry shape, dtype, and device information.
- Side-effectful or unsupported operations must trigger immediate
  materialization or explicit failure.
- The system must distinguish view-like operations from allocating operations.

## Materialization Rules

Materialization should occur when:

- a value is read by host code,
- an unsupported operation is encountered,
- explicit `realize()` is called,
- the system crosses an eager-only API boundary,
- debug mode requests immediate validation.

The RFC should define whether materialization occurs per value, per subgraph, or
per capture session. The recommended approach is subgraph-level realization with
internal scheduling opportunities.

## Interaction with Autograd

Autograd must remain correct under lazy execution. The design must specify:

- whether forward lazy graphs capture backward metadata directly,
- when backward graphs are materialized,
- how saved tensors are represented,
- how graph lifetime interacts with gradients and optimizer steps.

## API Direction

Possible public surface:

- scoped lazy context,
- explicit `toLazy()` conversion,
- explicit `realize()` and graph inspection hooks,
- build or runtime flag for debug validation.

The exact API should stay minimal until implementation experience is gathered.

## Work Breakdown

### Workstream A: Representation

- define lazy tensor and node types,
- record operation attributes and dependencies,
- carry metadata and source annotations.

### Workstream B: Capture Rules

- eager-to-lazy boundaries,
- view semantics,
- unsupported op handling,
- autograd capture interactions.

### Workstream C: Realization

- scheduler boundary,
- backend lowering hook,
- debug graph dumps and tracing.

### Workstream D: Validation

- correctness tests against eager mode,
- benchmark comparisons on representative workloads,
- memory pressure tests.

## Testing Plan

- eager-vs-lazy parity tests,
- autograd correctness tests,
- shape and dtype propagation tests,
- partial realization tests,
- benchmark suite coverage for capture cost and realized execution.

## Acceptance Criteria

- Lazy mode is opt-in and does not change eager semantics.
- Captured graphs can be materialized deterministically.
- Autograd remains correct on supported lazy workloads.
- At least one reference model sees measurable optimization opportunity from
  lazy capture plus later passes.

## Risks

- Hidden materialization can make performance hard to reason about.
- Lifetime bugs may appear if saved tensors outlive their graph session.
- The API can become too magical if lazy/eager boundaries are not explicit.

## Open Questions

- Do we model lazy tensors as a separate type or a mode on existing tensors?
- Should shape inference happen eagerly at capture time or lazily during lower?
- Can the existing graph manager participate directly, or do we need a new IR?

