# RFC-0006: Lazy Tensors

Status: `Ready`
Priority: `P1`
Depends on: RFC-0001, RFC-0002, RFC-0003
Blocks: RFC-0007, RFC-0008, RFC-0010
Last updated: `2026-03-30`

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

## Agentic Context

### Consolidated Status (2026-03-30)

**Landed:**
- Observe-mode capture with scoped sessions, tensor records, text/D2/JSON
  graph dumps, materialization events, and structured op attributes
  (`src/lazy.zig`, `src/ndtensor.zig`, `src/nn/nn.zig`, `src/nn/loss.zig`)
- Deferred forward execution via thunk queue with auto-realize at
  `realize()` and `copy_to_host()` boundaries
  (`src/lazy.zig`, `src/device/device_reference.zig`, `src/ndtensor.zig`)
- 8 regression tests covering capture, attributes, JSON, deferred parity,
  auto-realize, matmul chains, and metadata correctness

**Not yet landed:**
- Deferred backward pass (autograd ops in deferred mode) — see
  [`docs/next-milestones.md`](../next-milestones.md) M-6
- Subgraph-level realization (current flush is global FIFO)
- Backend-specific deferred batching and scheduling

**Key files:** `src/lazy.zig` (session + thunk infra), `src/device/device_reference.zig`
(dispatch interception + DeferredDispatchThunk), `src/ndtensor.zig` (capture hooks +
realize/host-read flush)

**Downstream consumers:** `src/graph_ir.zig` (RFC-0007) lowers session captures
into a typed graph IR for optimization.

---

### 2026-03-29 Opt-In Lazy Capture Session Foundation

- Completed:
  - Added a first-class `zg.lazy` capture surface in
    [`src/lazy.zig`](../../src/lazy.zig)
    and exported it through
    [`src/zigrad.zig`](../../src/zigrad.zig),
    including scoped session guards, stable tensor records, text plus D2 graph
    dumps, and recorded materialization events.
  - Instrumented generic tensor construction in
    [`src/ndtensor.zig`](../../src/ndtensor.zig)
    so eager tensor constructors and dependent ops now feed the lazy session
    without changing eager execution semantics. The landed slice records dtype,
    shape, device, storage kind, labels, parent edges, and whether tensors are
    attached or require gradients.
  - Added an explicit `NDTensor.realize()` boundary and host-read capture hooks
    so RFC-0006 now has a concrete, testable materialization API even though
    execution is still eager under the hood.
  - Added regression coverage in
    [`src/ndtensor.zig`](../../src/ndtensor.zig)
    for in-session capture, external-input capture when tensors predate the
    capture region, view recording via `alias()`, D2/text dump emission, and
    explicit versus host-read materialization events.
- Remains:
  - Introduce a true lazy execution mode where captured tensors can defer
    backend work until `realize()` rather than only shadowing eager execution.
  - Define operation-attribute capture beyond the current op-name/shape/device
    metadata so RFC-0007 can lower the session records into a richer optimizer
    IR.
  - Decide whether the eventual public API uses a separate `LazyTensor` handle
    or continues with scoped capture on the existing tensor type.
- Blockers:
  - This slice intentionally preserves eager execution; no deferred scheduler,
    subgraph realization engine, or backend lowering exists yet, so RFC-0007
    remains blocked on additional RFC-0006 work.
- Validation performed:
  - `zig fmt src/lazy.zig src/ndtensor.zig src/zigrad.zig`
  - `zig build test`

### 2026-03-29 Operation Attributes and JSON Session Dumps

- Completed:
  - Extended
    [`src/lazy.zig`](../../src/lazy.zig)
    so lazy-session tensor records now carry structured operation attributes in
    addition to op names, dtype, shape, device, storage kind, labels, and
    parent edges.
  - Added a machine-readable JSON dump surface to the lazy session alongside
    the existing text and D2 writers, giving RFC-0006 a stable inspection
    format that RFC-0007 can consume in verifier and IR-lowering work.
  - Threaded representative op metadata through
    [`src/ndtensor.zig`](../../src/ndtensor.zig),
    [`src/nn/nn.zig`](../../src/nn/nn.zig), and
    [`src/nn/loss.zig`](../../src/nn/loss.zig),
    including reshape target shapes, transpose permutations, subset steps,
    matmul transpose/scale parameters, max-along reduction settings, transfer
    device transitions, and softmax/loss capture names instead of generic
    `"op"` placeholders.
  - Added regression coverage in
    [`src/ndtensor.zig`](../../src/ndtensor.zig)
    for attribute-rich lazy capture plus JSON serialization/parsing so the new
    metadata surface is verified end to end rather than only through manual
    debug dumps.
- Remains:
  - Introduce a true lazy execution mode where captured tensors can defer
    backend work until `realize()` rather than only shadowing eager execution.
  - Define how RFC-0007 should lower the new session metadata into a stable IR
    with verifier-friendly value/node identities instead of stopping at session
    dumps.
  - Expand attribute capture to more ops once optimizer work identifies which
    attributes are required beyond the current representative set.
- Blockers:
  - This slice still preserves eager execution underneath capture, so RFC-0007
    has richer metadata to build on but no deferred scheduler or executable IR
    lowering yet.
- Validation performed:
  - `zig fmt src/lazy.zig src/ndtensor.zig src/nn/nn.zig src/nn/loss.zig`
  - `zig build test`

### 2026-03-30 Deferred Execution via Thunk Queue

- Completed:
  - Introduced true deferred execution mode (`ExecutionMode.deferred`) in
    [`src/lazy.zig`](../../src/lazy.zig)
    where `Session.mode = .deferred` causes all `DeviceReference.dispatch()`
    calls to enqueue type-erased thunks instead of executing immediately.
  - Added `ThunkBase` vtable struct and comptime-generic
    `DeferredDispatchThunk` in
    [`src/device/device_reference.zig`](../../src/device/device_reference.zig)
    that captures opspec params, device pointer, and deep-copies metadata
    slices (`[]const usize` fields like bmm shape arrays) to prevent
    dangling references to caller stack frames.
  - `Session.flush()` replays all queued thunks in FIFO order, then frees
    thunk memory. `NDTensor.realize()` and `NDTensor.copy_to_host()` now
    auto-flush deferred thunks before accessing data, so users get correct
    results at materialization boundaries.
  - The default mode remains `.observe` (capture-only, eager execution), so
    all existing code paths are unchanged.
  - Added five deferred-mode regression tests in
    [`src/ndtensor.zig`](../../src/ndtensor.zig):
    eager-vs-deferred parity, auto-realize on host read, multi-op chain
    with transpose and matmul, observe-mode unchanged, and capture metadata
    alongside deferred thunks.
- Remains:
  - Deferred backward pass (autograd in deferred mode).
  - Subgraph-level realization optimization (current flush is global FIFO).
  - Backend-specific deferred batching and scheduling.
  - Define how RFC-0007 should consume deferred session state for IR
    lowering and verification.
- Blockers:
  - Deferred execution is forward-only for now; backward pass still requires
    eager materialization before `backward()`.
  - RFC-0007 now has a working deferred execution surface to build on but
    still needs a formal IR and pass infrastructure.
- Validation performed:
  - `zig fmt src/lazy.zig src/ndtensor.zig src/device/device_reference.zig`
  - `zig build test` — all tests pass including five new deferred-mode tests
  - `zig build test-example-smoke` — no regressions
  - `zig build test-provider-parity` — no regressions
