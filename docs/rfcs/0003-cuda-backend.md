# RFC-0003: CUDA Backend

Status: `Ready`  
Priority: `P0`  
Depends on: RFC-0001  
Blocks: RFC-0006, RFC-0012  
Last updated: `2026-03-28`

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

- benchmark coverage through RFC-0001, including backend-aware checked-in CUDA
  benchmark specs and explicit non-CUDA skip semantics,
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

## Agentic Context

### 2026-03-28 Device-Safe Loss Fallbacks For Maintained Workloads

- Completed:
  - Reworked
    [`src/nn/loss.zig`](../../src/nn/loss.zig)
    so `softmax_cross_entropy_loss`, `softmax`, `smooth_l1_loss`, and
    `mse_loss` no longer dereference off-host tensor storage directly from Zig
    loops. The host path stays direct, while non-host devices now stage tensors
    through explicit host copies before computing the reference loss or
    gradient update.
  - Added a regression test in
    [`src/nn/tests/test_loss.zig`](../../src/nn/tests/test_loss.zig)
    covering `softmax` over a non-last dimension, which also validates the
    stride-aware host reference path reused by the new off-host fallback.
  - Revalidated the maintained smoke portfolio through `zig build test`, so
    the MNIST, DQN, and GCN training examples still run after the loss-layer
    change.
- Remains:
  - Replace the host-staged fallback implementations with dedicated
    device-native loss kernels once the device API and CUDA kernel surface are
    ready for that narrower optimization pass.
  - Execute the same maintained workloads on a real CUDA-capable host to prove
    the new fallback path behaves correctly under actual device allocations and
    stream ordering.
- Blockers:
  - This environment still exposes no CUDA toolkit or CUDA device, so the new
    loss-path safety work was validated through host tests and smoke runs
    rather than accelerator execution.
- Validation performed:
  - `zig build test`

### 2026-03-28 Benchmark Harness CUDA Integration

- Completed:
  - Extended the RFC-0001 benchmark harness through
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/result.zig`](../../benchmarks/src/result.zig),
    [`benchmarks/src/metadata.zig`](../../benchmarks/src/metadata.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig),
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig), and
    [`benchmarks/src/validate.zig`](../../benchmarks/src/validate.zig)
    so checked-in benchmark specs can target `cuda[:index]`, successful CUDA
    runs surface structured device metadata, and non-CUDA environments emit
    explicit schema-valid `skipped` rows instead of aborting the harness.
  - Added checked-in CUDA-targeted model benchmark specs in
    [`benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json`](../../benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json)
    and
    [`benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json`](../../benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json),
    plus
    [`tests/src/benchmark_cuda_request_smoke_main.zig`](../../tests/src/benchmark_cuda_request_smoke_main.zig)
    and `zig build test-benchmark-cuda-request-smoke`,
    so RFC-0003 now has a validated benchmark-contract surface even before real
    GPU hardware is available.
- Remains:
  - Execute the new CUDA-targeted benchmark specs on a real toolkit/device host
    and collect the first non-skipped CUDA model-train/model-infer artifacts.
  - Extend the same harness integration to future CUDA-specific primitive and
    memory suites once those workloads are validated on hardware.
- Blockers:
  - This macOS host still had no CUDA toolkit or CUDA device available, so the
    new benchmark slice validated only the explicit skip path and JSONL
    contract checks rather than real CUDA execution.
- Validation performed:
  - `zig build test-benchmark-cuda-request-smoke`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json --output .zig-cache/benchmark-cuda-spec.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/benchmark-cuda-spec.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json --baseline pytorch --output .zig-cache/benchmark-cuda-pytorch.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/benchmark-cuda-pytorch.jsonl`
  - `zig build test`

### 2026-03-28 Backend-Dispatched Adam Updates

- Completed:
  - Added a fused `adam` backend op through
    [`src/device/opspec.zig`](../../src/device/opspec.zig),
    [`src/device/host_device.zig`](../../src/device/host_device.zig),
    [`src/device/cuda_device.zig`](../../src/device/cuda_device.zig),
    [`src/cuda/blas_conflux.h`](../../src/cuda/blas_conflux.h),
    [`src/cuda/blas_conflux.cu`](../../src/cuda/blas_conflux.cu),
    and
    [`src/cuda/blas/adam.cu`](../../src/cuda/blas/adam.cu),
    so dense Adam updates no longer require host-side loops and the CUDA
    backend has an optimizer update primitive that matches the host path.
  - Fixed
    [`src/nn/optim.zig`](../../src/nn/optim.zig)
    so Adam increments its bias-correction timestep once per logical
    `Optimizer.step()` instead of once per parameter, removing a correctness
    bug that affected multi-parameter models such as DQN and GCN.
  - Removed the explicit non-host panic from Adam updates in
    [`src/nn/optim.zig`](../../src/nn/optim.zig),
    which closes a concrete CUDA training blocker for the maintained example
    models once the runtime is available.
  - Added unit coverage in
    [`src/nn/optim.zig`](../../src/nn/optim.zig)
    asserting repeated optimizer steps share a single Adam timestep across all
    attached parameters.
- Remains:
  - Compile and execute the new CUDA Adam kernel on a real toolkit/device
    combination.
  - Add dedicated optimizer parity or example-level CUDA smoke coverage on
    GPU-capable infrastructure.
- Blockers:
  - This macOS host still had no CUDA toolkit or CUDA device available, so the
    new fused optimizer path validated only through host builds/tests and code
    integration rather than a real CUDA compile/run.
- Validation performed:
  - `zig build test`

### 2026-03-28 CUDA Example Path Audits

- Completed:
  - Made [`src/ndarray.zig`](../../src/ndarray.zig) gather offset computation
    device-safe by staging index offsets on the host before transferring them
    into device memory, removing a direct host write into CUDA buffers.
  - Added CUDA implementations for `scatter_add`,
    `scatter_gcn_deg_scaled`, and `scatter_gcn_deg_scaled_bwd` in
    [`src/cuda/cuda_utils.cu`](../../src/cuda/cuda_utils.cu),
    declared through [`src/cuda/cuda_utils.h`](../../src/cuda/cuda_utils.h),
    and wired them into
    [`src/device/cuda_device.zig`](../../src/device/cuda_device.zig) so DQN
    gather backprop and the GCN message-passing kernel surface no longer panic
    on CUDA dispatch.
  - Enabled CUDA runtime selection in
    [`examples/dqn/src/dqn_train.zig`](../../examples/dqn/src/dqn_train.zig)
    and [`examples/gcn/src/main.zig`](../../examples/gcn/src/main.zig).
  - Removed the remaining GCN host-view assumptions by copying masks and eval
    logits to host explicitly in
    [`examples/gcn/src/model.zig`](../../examples/gcn/src/model.zig) and
    [`examples/gcn/src/main.zig`](../../examples/gcn/src/main.zig), and made
    the synthetic smoke dataset in
    [`examples/gcn/src/dataset.zig`](../../examples/gcn/src/dataset.zig) use a
    non-prefix training mask so the host smoke path validates the corrected
    mask mapping.
- Remains:
  - Validate the CUDA runtime path on a real toolkit/device combination.
  - Add dedicated CUDA smoke and benchmark runs once GPU-capable runners are
    available.
- Blockers:
  - This macOS host still had no CUDA toolkit or CUDA device available, so the
    new kernels and runtime selectors validated only through host builds/tests
    and code-path audit rather than a real CUDA compile/run.
- Validation performed:
  - `zig build test`
  - `cd examples/dqn && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `cd examples/gcn && ZG_EXAMPLE_SMOKE=1 zig build run`

### 2026-03-28 Runtime Device Selection + CUDA Diagnostics

- Completed:
  - Added shared runtime-device selection in
    [`src/device/runtime_device.zig`](../../src/device/runtime_device.zig),
    exported through [`src/device.zig`](../../src/device.zig) and
    [`src/zigrad.zig`](../../src/zigrad.zig), so callers can request
    `ZG_DEVICE=host|cpu|cuda[:index]` with explicit build-time and availability
    errors.
  - Added public CUDA diagnostics in
    [`src/device/cuda_device.zig`](../../src/device/cuda_device.zig) covering
    selected device id, device name, compute capability, multiprocessor count,
    total global memory, and CUDA driver/runtime versions, plus the
    `ZG_CUDA_DIAGNOSTICS=1` opt-in logging hook.
  - Fixed CUDA context teardown in
    [`src/cuda/cuda_utils.cu`](../../src/cuda/cuda_utils.cu) and
    [`src/cuda/device_properties.cu`](../../src/cuda/device_properties.cu) by
    surfacing `deinit_device(...)` and destroying the owned `CUcontext` during
    `CudaDevice.deinit()`.
  - Updated standalone example build scripts under [`examples/`](../../examples/)
    to accept `-Denable_cuda=true` and `-Drebuild_cuda=true`, and wired the
    hello-world plus MNIST entrypoints to the shared selector.
  - Added host-copy helpers in [`src/ndtensor.zig`](../../src/ndtensor.zig) so
    the MNIST evaluation path no longer assumes host-backed tensor storage when
    reading prediction batches.
- Remains:
  - Validate the CUDA runtime path on a real toolkit/device combination.
  - Add dedicated CUDA smoke and benchmark runs once GPU-capable runners are
    available.
- Blockers:
  - This macOS host had no CUDA toolkit or CUDA device available, so the run
    validated only the non-CUDA build path plus the new explicit error/reporting
    surfaces.
- Validation performed:
  - `zig build test`
  - `cd examples/hello-world && zig build run`
  - `cd examples/hello-world && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
  - `cd examples/dqn && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
  - `cd examples/gcn && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
