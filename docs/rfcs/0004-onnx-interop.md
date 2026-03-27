# RFC-0004: ONNX Interoperability

Status: `Planned`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0007  
Blocks: RFC-0010  
Last updated: `2026-03-27`

## Summary

This RFC defines ONNX import and export support for Zigrad. The initial focus is
practical interoperability for common inference and training-adjacent graphs,
with a clear operator support matrix, graph validation rules, and a staged plan
for importer-first delivery.

## Motivation

ONNX is the most common interchange format requested by users moving models
between frameworks, serving stacks, and compilers. Without ONNX support, Zigrad
remains isolated from external tooling and harder to evaluate in heterogeneous
workflows.

This RFC is deliberately placed after static graph work because reliable ONNX
import and export become far easier once Zigrad can represent graph programs in
a stable internal IR instead of only as eager runtime actions.

## Goals

- Import a useful ONNX subset into Zigrad graph IR.
- Export a useful Zigrad subset back to ONNX.
- Publish an operator support matrix and versioning policy.
- Preserve enough source metadata for debugging unsupported nodes.
- Support both eager execution materialization and future compiled flows.

## Non-Goals

- Full ONNX operator parity in the first release.
- Immediate support for every training-oriented ONNX operator.
- Supporting custom ONNX dialects or vendor-specific extensions initially.
- Replacing native Zigrad checkpoint formats.

## Scope

Initial ONNX support will target:

- tensors, constants, initializers,
- shape metadata and dtypes,
- linear algebra operations,
- common elementwise ops,
- reductions,
- activations,
- reshape/transpose/concat/slice patterns,
- limited convolutional graphs needed by reference examples.

Training graph export may be phased behind inference import/export.

## Architecture

Introduce `src/interop/onnx/` with:

- parser and protobuf binding layer,
- model validation layer,
- attribute and tensor decoding utilities,
- ONNX-to-Zigrad graph lowering,
- Zigrad-to-ONNX exporter,
- operator registry and capability matrix.

The operator registry must separate:

- `supported`,
- `supported-with-constraints`,
- `unsupported`.

## Import Pipeline

1. Parse ONNX model and metadata.
2. Validate graph well-formedness and opset constraints.
3. Decode initializers and constants.
4. Lower nodes into Zigrad graph IR.
5. Apply import-time canonicalization for equivalent patterns.
6. Emit a graph plus import diagnostics.

Diagnostics must include original node name, op type, and unsupported attribute
information.

## Export Pipeline

1. Start from validated Zigrad graph IR.
2. Normalize graph into export-friendly patterns.
3. Map operations and attributes into ONNX equivalents.
4. Emit initializers and value info.
5. Validate the exported model with an ONNX checker where feasible.

## Operator Support Policy

The project must publish a machine-readable support table with:

- ONNX op name,
- supported opset range,
- Zigrad lowering target,
- known limitations,
- test coverage status.

Unsupported operators must fail loudly with actionable diagnostics.

## Data Type and Shape Policy

- `f32` and `f64` are required in the first milestone.
- integer support should cover the needs of indexing and metadata operations.
- symbolic shapes may be parsed but need not be fully executable at first.
- export must not silently erase dynamic-shape semantics.

## Testing Plan

- unit tests for tensor, attribute, and initializer decoding,
- import tests for curated ONNX fixtures,
- export round-trip tests for supported graph subsets,
- execution parity tests against reference runtimes where practical,
- benchmark coverage for large model import latency.

## Milestones

### Milestone A: Importer MVP

- parse ONNX models,
- lower a supported operator subset,
- execute imported inference graphs.

### Milestone B: Exporter MVP

- export supported Zigrad graphs,
- validate with ONNX tooling,
- round-trip on representative examples.

### Milestone C: Coverage Expansion

- broaden operator set,
- improve diagnostics,
- add model zoo fixtures.

## Acceptance Criteria

- At least one non-trivial ONNX model imports and executes correctly.
- Exported models validate for the supported subset.
- Unsupported operators report precise diagnostics.
- Operator support matrix is published and tested.

## Risks

- ONNX operator semantics can differ subtly from internal assumptions.
- Shape inference complexity can grow quickly if symbolic support is rushed.
- Exporting dynamic or lazily-captured graphs may require more normalization than
  expected.

## Open Questions

- Should the importer target eager execution directly or only graph IR?
- Do we vendor generated ONNX protobuf types or keep parsing isolated?
- When do we support training graph export versus inference-only export?

