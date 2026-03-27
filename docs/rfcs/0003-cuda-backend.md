# RFC-0003: CUDA Backend

Status: `Ready`  
Priority: `P0`  
Depends on: RFC-0001  
Blocks: RFC-0006, RFC-0012  
Last updated: `2026-03-27`

## Summary

This RFC defines the work required to move CUDA support from experimental to a
supported backend for core training and inference workloads. It covers device
discovery, memory management, stream semantics, kernel coverage, autograd
integration, build ergonomics, testing, and performance validation.

## Motivation

The README explicitly calls CUDA support experimental. That is incompatible with
the roadmap items around benchmarking, graph optimization, compilers, LLM
examples, and inference translation. A serious GPU story is prerequisite for
nearly every advanced capability the project aims to ship.

The existing device proposal in
[docs/2025-03-22-device-proposal.md](../2025-03-22-device-proposal.md)
should inform how operation dispatch evolves while this backend matures.

## Goals

- Make CUDA a supported backend for representative research workloads.
- Define a stable memory and stream model for tensors and operations.
- Cover the kernel surface needed by current examples and near-term RFCs.
- Integrate profiling and benchmark visibility from day one.
- Preserve eager usability while preparing for lazy and compiled execution.

## Non-Goals

- Multi-GPU distributed training.
- NCCL integration.
- Kernel autotuning in the first milestone.
- Full parity with every CPU-only utility kernel.

## Backend Contract

The CUDA backend must provide:

- device enumeration and capability reporting,
- explicit memory ownership and lifetime rules,
- async stream-aware execution,
- synchronization primitives suitable for debugging and correctness tests,
- error mapping from CUDA/cuBLAS/cuDNN APIs into Zigrad errors,
- transfer utilities between host and device tensors.

## Operator Coverage Targets

Initial supported operator categories:

- tensor creation and movement,
- pointwise arithmetic,
- reductions,
- indexing and gather required by DQN-like workloads,
- GEMM and batched GEMM,
- common activation functions,
- convolution support required by current and near-term examples,
- backward kernels needed by the supported forward set.

## Memory Model

The backend must define:

- owned device buffers,
- borrowed views where valid,
- allocator and pool semantics,
- host-pinned transfer support where it materially improves throughput,
- explicit synchronization boundaries for materialization and debugging.

Memory APIs must be benchmarked and leak-tested under repeated training loops.

## Stream Model

The backend will start with one explicit default stream per execution context,
while keeping the design extensible to multiple streams later. The model should
support:

- enqueueing kernels and copies,
- explicit synchronize points,
- future capture for lazy or compiled execution,
- optional debug mode that synchronizes aggressively.

## Build and Tooling

- CUDA builds must be explicitly enabled.
- Missing toolkit components must produce actionable build errors.
- Backend metadata must expose CUDA toolkit and driver versions.
- CI should at least build the CUDA path even if full GPU execution runs on
  scheduled or dedicated hardware.

## Work Breakdown

### Workstream A: Runtime Foundation

- device discovery,
- backend metadata,
- allocator/pool implementation,
- copy and synchronization utilities.

### Workstream B: Operator Parity

- pointwise kernels,
- reductions,
- GEMM bindings,
- gather/indexing,
- convolution-critical kernels.

### Workstream C: Autograd Integration

- backward kernel registration,
- gradient accumulation semantics,
- correctness tests against CPU baselines.

### Workstream D: Observability and Validation

- benchmark coverage through RFC-0001,
- profiling hooks for kernel and transfer timing,
- integration tests on representative examples.

## Testing Plan

- Device smoke tests for initialization and teardown.
- Memory leak and double-free stress tests.
- CPU-vs-CUDA numerical parity tests.
- Example-level tests for MNIST and DQN.
- Benchmark-based validation on real GPU hardware.

## Acceptance Criteria

- CUDA backend reports device metadata and initializes reliably.
- Core example workloads run end-to-end on CUDA.
- No known lifetime bugs remain in repeated train/eval loops.
- Benchmark results exist for at least one model training and one inference
  workload on CUDA.
- README can remove the "experimental" label after sustained validation.

## Risks

- Kernel surface growth may outpace validation capacity.
- Async execution can hide correctness bugs if synchronization policy is vague.
- Build friction may remain high across developer environments.

## Follow-On Work

- Multi-stream scheduling once lazy tensors exist.
- NCCL/GPU distributed work after the single-device path is stable.
- Kernel autotuning or TVM-assisted generation later in the roadmap.
