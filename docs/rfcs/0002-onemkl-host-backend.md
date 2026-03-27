# RFC-0002: oneMKL Host Backend

Status: `Ready`  
Priority: `P0`  
Depends on: RFC-0001  
Blocks: RFC-0006, RFC-0012  
Last updated: `2026-03-28`

## Summary

This RFC defines a production-grade CPU backend strategy centered on explicit
provider selection between Accelerate, OpenBLAS, and oneMKL, with oneMKL as the
highest-performance portable CPU target for x86 deployments. The work covers
build integration, capability discovery, provider-specific dispatch, coverage of
 critical linear algebra paths, and benchmark-backed performance validation.

## Motivation

Zigrad already demonstrates strong Apple Silicon performance, but roadmap goals
require a broader CPU story that is:

- competitive on x86 systems,
- explicit about provider capabilities,
- testable across providers,
- decoupled from hard-coded assumptions in model code.

The existing notes in
[docs/roadmap.norg](../roadmap.norg)
also call out oneMKL and eventual oneDNN work. This RFC scopes the first part:
robust host BLAS/LAPACK acceleration with oneMKL support.

## Goals

- Support explicit CPU provider selection at build time.
- Preserve a stable high-level tensor API above provider details.
- Cover the operator families that dominate current workloads.
- Make provider capability differences visible through backend metadata.
- Ensure benchmark and correctness parity across Accelerate, OpenBLAS, and
  oneMKL where supported.

## Non-Goals

- Integrating oneDNN in this RFC.
- Rewriting the entire host backend around oneMKL-specific APIs.
- Supporting every LAPACK routine on day one.
- Distributed CPU execution.

## Scope

This RFC covers:

- GEMM, GEMV, batched GEMM where available,
- convolution helper paths that depend on GEMM,
- reductions or utility kernels that remain custom but run beside BLAS,
- thread and affinity configuration,
- provider selection and diagnostics.

## Proposed Architecture

Introduce a `HostBlasProvider` enum and explicit backend configuration:

```zig
pub const HostBlasProvider = enum {
    accelerate,
    openblas,
    mkl,
};
```

The host backend will expose:

- provider metadata,
- supported dtype matrix,
- supported op families,
- thread configuration,
- fallback reason reporting.

Provider selection must be explicit in the build graph and discoverable at
runtime through backend inspection and benchmark metadata.

## Build and Packaging Requirements

- `build.zig` must expose host backend options without leaking provider-specific
  headers into unrelated builds.
- oneMKL include/library path configuration must be documented and overridable.
- Provider-specific builds must fail clearly when headers or libraries are
  missing.
- CI should build at least one non-Accelerate host backend configuration.

## Execution Model

The host backend should preserve the same higher-level tensor semantics
regardless of provider:

- contiguous and non-contiguous input handling must be defined,
- dtype promotion rules must remain framework-level, not provider-level,
- errors such as unsupported dtype or transpose mode must surface through
  backend diagnostics rather than silent fallback.

## Work Breakdown

### Workstream A: Provider Abstraction

- [x] Formalize provider selection.
- [x] Introduce runtime-visible provider metadata.
- [x] Separate capability discovery from operation dispatch for the current host
  BLAS selection path.

### Workstream B: Kernel Coverage

- [x] GEMM and GEMV for `f32`, `f64`.
- [x] Batched GEMM with correct nested batch-broadcast semantics, preserving
  direct batched dispatch for modulo-safe layouts.
- [ ] Audit conv and linear layers to ensure provider-backed paths are taken.

### Workstream C: Correctness and Debuggability

- Cross-provider test suite with numeric tolerances.
- Logging hooks to record selected provider and fallback mode.
- Reference fallback path for unsupported cases.

### Workstream D: Performance Validation

- Benchmark against PyTorch CPU.
- Publish provider comparison tables for representative models.
- Validate thread scaling behavior.

## Testing Plan

- Unit tests for provider selection and capability reporting.
- Cross-provider numerical equivalence tests.
- Integration tests for MNIST, GCN, and DQN CPU execution.
- Benchmark comparisons captured through RFC-0001.

## Acceptance Criteria

- A user can build with Accelerate, OpenBLAS, or oneMKL explicitly.
- CPU backend metadata identifies the active provider.
- Core training and inference examples run correctly on the oneMKL path.
- Benchmark results show no regression relative to the current default provider
  on existing supported workloads.

## Risks

- oneMKL packaging friction can slow adoption.
- Provider feature mismatch may force more fallback complexity than expected.
- Thread oversubscription can distort benchmark outcomes.

## Follow-On Work

- oneDNN integration once the provider abstraction is stable.
- provider-specific fused kernels where BLAS alone is insufficient.

## Agentic Context

### 2026-03-27 Provider Selection + Metadata

- Completed:
  - Added [`src/device/host_blas_provider.zig`](../../src/device/host_blas_provider.zig)
    to formalize `HostBlasProvider` and `HostBackendInfo`.
  - Replaced the build-graph `-Denable_mkl` toggle with explicit
    `-Dhost_blas=auto|accelerate|openblas|mkl` selection while keeping
    `-Denable_mkl=true` as a compatibility alias for `mkl`.
  - Added documented oneMKL include/library overrides in `build.zig` via
    `-Dmkl_include_dir` and `-Dmkl_library_dir`.
  - Exposed the configured provider through the host backend and benchmark
    metadata so JSONL results now report `accelerate`, `openblas`, or `mkl`
    instead of the ambiguous Linux `blas` label.
  - Updated repository docs and Linux CI entrypoints to use explicit host BLAS
    provider selection.
- Remains:
  - Validate OpenBLAS and oneMKL builds on Linux/x86 hardware and add
    cross-provider numerical parity coverage.
  - Audit conv, linear, and batched-GEMM call sites so provider-backed paths
    are explicitly covered by tests and benchmarks.
  - Add provider comparison benchmark tables once Linux OpenBLAS and oneMKL
    environments are available.
- Blockers:
  - This run validated only the macOS Accelerate path locally. OpenBLAS and
    oneMKL configuration was implemented but not executed in this environment.
- Validation performed:
  - `zig build test`
  - `zig build -Dhost_blas=accelerate benchmark`
  - `python3 - <<'PY'`
    `import json`
    `from pathlib import Path`
    `first = json.loads(Path("benchmarks/results/latest.jsonl").read_text().splitlines()[0])`
    `print(first["backend"]["host_provider"])`
    `PY`

### 2026-03-28 Batched GEMM Broadcast Correctness

- Completed:
  - Fixed batched matmul broadcast indexing in
    [`src/ndarray.zig`](../../src/ndarray.zig) so nested batch broadcasts such
    as `[2,2,...] x [2,1,...]` now match PyTorch-style semantics instead of the
    previous flatten-and-modulo shortcut.
  - Added a generic per-batch `matmul` fallback for non-modulo-safe broadcast
    layouts while preserving the existing direct batched dispatch for common
    safe layouts, including linear-layer style `[batch,...] x [weight]` cases.
  - Fixed accumulation into broadcast-compatible outputs in `bmm_acc_`, which
    is required for backward passes that reduce repeated batch contributions
    into a smaller gradient tensor.
  - Updated [`src/ndtensor.zig`](../../src/ndtensor.zig) so batched matmul
    options forward `alpha`/`beta`, making the existing backward accumulation
    call sites behave as intended.
  - Added nested broadcast forward and backward regression tests in
    [`src/ndarray.zig`](../../src/ndarray.zig) and
    [`src/ndtensor.zig`](../../src/ndtensor.zig).
  - Updated the example build entrypoints under [`examples/`](../../examples/)
    to accept `-Dhost_blas=...` directly while preserving `-Denable_mkl=true`
    as a local compatibility alias.
  - Updated [`examples/gcn/src/main.zig`](../../examples/gcn/src/main.zig) to
    use the current `std.json.Stringify` API so the GCN example builds again on
    the current Zig toolchain.
- Remains:
  - Validate OpenBLAS and oneMKL builds on Linux/x86 hardware and add
    cross-provider numerical parity coverage.
  - Audit conv and linear call paths beyond the matmul broadcast fix and add
    example/runtime smoke coverage for the provider-sensitive paths.
  - Add runtime smoke coverage for the example portfolio now that the explicit
    `host_blas` entrypoints are aligned.
- Blockers:
  - This run still had no Linux OpenBLAS/oneMKL environment, so provider parity
    remains unexecuted locally.
- Validation performed:
  - `zig build test`
  - `cd examples/hello-world && zig build -Dhost_blas=accelerate`
  - `cd examples/mnist && zig build -Dhost_blas=accelerate`
  - `cd examples/dqn && zig build -Dhost_blas=accelerate`
  - `cd examples/gcn && zig build -Dhost_blas=accelerate`
