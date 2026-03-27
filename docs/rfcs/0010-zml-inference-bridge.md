# RFC-0010: ZML Inference Bridge

Status: `Draft`  
Priority: `P2`  
Depends on: RFC-0007  
Blocks: None  
Last updated: `2026-03-27`

## Summary

This RFC defines a translation path from Zigrad-authored or Zigrad-imported
models into ZML for inference-only execution. The goal is to preserve Zigrad as
the authoring, experimentation, or training environment while enabling a clean
handoff to a dedicated inference runtime when appropriate.

## Motivation

The existing roadmap notes suggest that pure inference mode may eventually
defer to ZML. This is a strong strategic fit: Zigrad focuses on flexible
research and training, while ZML can serve as a specialized inference target.
To make that real, the bridge must be explicit, testable, and constrained.

## Goals

- Translate a supported Zigrad graph subset into a ZML-compatible form.
- Keep the bridge inference-only in the initial version.
- Surface unsupported operations clearly.
- Reuse static graph optimization outputs rather than lowering raw eager code.
- Benchmark translation cost and inference performance.

## Non-Goals

- Training execution on ZML.
- Automatic, silent backend switching inside user code.
- Translating arbitrary dynamic control flow in the first milestone.
- Replacing native Zigrad inference for all users.

## Translation Boundary

The bridge should consume a verified, optimized static graph from RFC-0007. It
must not depend on a specific front-end path such as ONNX import only. Supported
sources should include:

- native Zigrad models after graph capture,
- imported ONNX graphs once available,
- GGUF-backed model graphs where representable.

## Supported Model Classes

The first milestone should prioritize:

- feed-forward networks,
- convolutional inference graphs,
- transformer encoder/decoder blocks needed by the first LLM example subset.

## Architecture

Introduce `src/interop/zml/` with:

- capability checker,
- graph lowering and op mapping,
- parameter/tensor conversion,
- packaging or handoff utilities,
- execution comparison test helpers.

The bridge must generate diagnostics that identify:

- unsupported ops,
- unsupported dtypes,
- unsupported layouts,
- fallback recommendations.

## Translation Policy

The lowering pipeline must preserve:

- shape semantics,
- dtype semantics,
- parameter names where useful,
- deterministic value ordering.

Lossy lowering must be forbidden unless explicitly opted into.

## Benchmarking Requirements

Benchmarks must compare:

- native Zigrad inference,
- optimized Zigrad inference,
- translated ZML inference,
- translation overhead amortization.

Results should guide whether this bridge is a niche utility or a primary
deployment path.

## Work Breakdown

### Workstream A: Capability Matrix

- enumerate ZML-supported op subset,
- map Zigrad graph nodes into support categories,
- define failure diagnostics.

### Workstream B: Lowering and Packaging

- lower supported graphs,
- convert parameters and constants,
- emit runnable artifacts.

### Workstream C: Validation

- parity tests between native and translated inference,
- benchmark coverage on at least one CV and one transformer-style graph.

## Acceptance Criteria

- At least one non-trivial model translates successfully.
- Native and translated inference match numerically within declared tolerances.
- Unsupported graphs fail with explicit diagnostics.
- Benchmarks quantify translation cost and inference tradeoffs.

## Risks

- ZML and Zigrad semantic mismatches may force more graph normalization than
  expected.
- The supported subset may initially be too narrow to justify maintenance cost.
- Packaging and dependency management could add operational complexity.

## Open Questions

- Should the bridge emit standalone artifacts or call ZML APIs directly?
- Do we require static shapes initially?
- How do we represent custom or fused operations that exist in Zigrad only?

