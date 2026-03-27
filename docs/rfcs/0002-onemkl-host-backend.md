# RFC-0002: oneMKL Host Backend

Status: `Ready`  
Priority: `P0`  
Depends on: RFC-0001  
Blocks: RFC-0006, RFC-0012  
Last updated: `2026-03-27`

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

- Formalize provider selection.
- Introduce runtime-visible provider metadata.
- Separate capability discovery from operation dispatch.

### Workstream B: Kernel Coverage

- GEMM and GEMV for `f32`, `f64`.
- Batched GEMM if supported efficiently by the provider.
- Audit conv and linear layers to ensure provider-backed paths are taken.

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
