# RFC-0009: MLIR Lowering Pipeline

Status: `Exploratory`  
Priority: `P2`  
Depends on: RFC-0007, RFC-0008  
Blocks: RFC-0011  
Last updated: `2026-03-27`

## Summary

This RFC defines an optional MLIR lowering pipeline for Zigrad graph IR. The
goal is to expose a structured compiler interchange and transformation layer
without forcing the entire project to become MLIR-first.

## Motivation

MLIR can provide:

- a rich ecosystem of optimization and lowering passes,
- interoperability with existing compiler infrastructure,
- a clearer bridge to external code generation paths.

However, MLIR also introduces major complexity and toolchain surface. This RFC
therefore treats it as optional and exploratory until Zigrad's own graph IR and
dynamic compiler story are stable.

## Goals

- Define a lowering from Zigrad graph IR into an MLIR representation.
- Identify the subset of MLIR dialects needed for initial experimentation.
- Keep MLIR integration optional at build time.
- Enable round-trip debugging and validation between internal IR and MLIR.

## Non-Goals

- Replacing the core internal IR with MLIR.
- Requiring MLIR for normal users or standard builds.
- Committing to a custom MLIR dialect unless necessary.
- Solving every backend lowering through MLIR in the first milestone.

## Candidate Dialects

Initial exploration should focus on:

- `arith`,
- `tensor`,
- `memref`,
- `linalg`,
- `scf`,
- `func`,
- `gpu` where relevant.

A custom `zg` dialect should only be introduced if existing dialects cannot
represent required semantics cleanly.

## Lowering Boundary

MLIR should consume verified, optimized graph IR, not raw eager operations. The
boundary should therefore sit after RFC-0007 and potentially after RFC-0008
normalization for compiled segments.

## Build and Tooling Strategy

- MLIR support must be feature-gated.
- The build should clearly report when MLIR tooling is unavailable.
- Generated MLIR should be inspectable via debug flags or file emission.
- CI can validate syntax or golden outputs even if full codegen is not run on
  every platform.

## Work Breakdown

### Workstream A: Representation Mapping

- map tensors, shapes, dtypes, and ops,
- define how layout, aliasing, and side effects are modeled,
- emit diagnostic metadata.

### Workstream B: Validation

- round-trip or equivalence testing where possible,
- golden MLIR dumps for representative graphs,
- verifier integration.

### Workstream C: Optimization Experiments

- identify passes that improve real workloads,
- compare internal optimizer output vs MLIR-assisted output,
- measure compile cost.

### Workstream D: Downstream Hooks

- prepare optional hooks for future LLVM, GPU, or TVM-adjacent paths,
- keep the surface modular.

## Testing Plan

- golden lowering tests,
- verifier tests on invalid graphs,
- benchmark coverage for lowering time,
- comparison tests between internal execution and MLIR-lowered execution where
  executable paths exist.

## Acceptance Criteria

- A supported subset of Zigrad IR lowers into valid MLIR.
- Lowered graphs are inspectable and tied back to source operations.
- The integration remains optional and does not burden non-MLIR users.
- At least one real optimization or downstream capability justifies the added
  complexity.

## Risks

- Toolchain complexity may slow contribution velocity.
- Dialect mismatches may push the project toward a custom dialect too early.
- The MLIR path may duplicate internal optimizer work with little payoff.

## Open Questions

- Do we start with textual MLIR emission only?
- Should the first milestone target CPU-only lowering?
- Is MLIR primarily a debug/introspection tool, or a production codegen path?

