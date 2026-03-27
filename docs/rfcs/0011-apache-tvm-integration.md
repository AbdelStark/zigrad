# RFC-0011: Apache TVM Integration

Status: `Exploratory`  
Priority: `P3`  
Depends on: RFC-0001, RFC-0007, RFC-0009  
Blocks: None  
Last updated: `2026-03-27`

## Summary

This RFC defines an optional integration path with Apache TVM for ahead-of-time
code generation, autotuning, and deployment-oriented optimization. TVM is not
the default execution path; it is an external compiler target to be justified by
measured wins on selected workloads.

## Motivation

TVM offers mature autotuning and deployment-oriented compilation across hardware
targets. If Zigrad can lower appropriate graph subsets into TVM-compatible
representations, it may gain access to optimized kernels or deployment routes
without having to reinvent every compiler technique internally.

That said, TVM also expands operational complexity significantly. Integration
must therefore be justified by benchmarks, isolated behind feature flags, and
kept optional.

## Goals

- Evaluate TVM as an optional target for optimized execution.
- Define a clear lowering boundary from Zigrad IR into TVM-facing IR.
- Enable benchmark-based comparison with native and MLIR-assisted paths.
- Keep the integration decoupled from the default user experience.

## Non-Goals

- Making TVM mandatory for compilation or deployment.
- Supporting every Zigrad operation through TVM.
- Replacing Zigrad's internal optimizer or runtime abstractions.
- Building distributed compilation infrastructure in the first milestone.

## Integration Options

Candidate strategies include:

- lower from Zigrad static graph IR directly into TVM-facing structures,
- lower through ONNX export and let TVM ingest ONNX,
- lower through MLIR if that path becomes mature enough.

This RFC should validate those paths rather than assuming the most complex one is
best.

## Evaluation Criteria

TVM integration is worth keeping only if it delivers one or more of:

- measurable inference speedups on selected models,
- access to hardware targets that Zigrad would otherwise not support soon,
- deployment or packaging benefits that users actually need.

## Work Breakdown

### Workstream A: Feasibility Spikes

- compare ONNX-based and direct-lowering approaches,
- identify the supported op subset,
- measure compile overhead.

### Workstream B: Prototype Lowering

- implement one lowering path,
- translate parameters and graph attributes,
- execute representative inference graphs.

### Workstream C: Validation

- correctness parity tests,
- benchmark comparisons against native execution,
- operational documentation.

## Testing Plan

- correctness tests on a small supported graph subset,
- integration tests for model packaging and execution,
- benchmark coverage for compile latency and steady-state throughput.

## Acceptance Criteria

- A prototype path exists from Zigrad into TVM for at least one meaningful
  workload.
- The benchmark program shows whether TVM produces a real benefit.
- The integration remains optional and isolated from the default build path.

## Risks

- Toolchain complexity may exceed the practical benefit.
- ONNX- or MLIR-mediated lowering can hide semantic mismatches.
- TVM support maintenance may become expensive if coverage remains narrow.

## Open Questions

- Is TVM best used through ONNX export, or should Zigrad lower directly?
- Which workload family should be the deciding benchmark: CV, transformers, or
  edge deployment models?
- Does TVM overlap too heavily with the planned MLIR and ZML work to justify
  itself?

