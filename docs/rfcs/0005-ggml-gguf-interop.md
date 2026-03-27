# RFC-0005: ggml and GGUF Interoperability

Status: `Planned`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0012  
Blocks: None  
Last updated: `2026-03-27`

## Summary

This RFC defines how Zigrad will interoperate with the ggml ecosystem,
especially GGUF checkpoints and metadata. The initial focus is loading common
GGUF assets into Zigrad for inference and analysis, followed by optional export
and partial quantization-aware execution support.

## Motivation

The roadmap explicitly calls out support for popular formats like ggml, and the
example roadmap includes LLM work. In practice, GGUF is the distribution format
users are likely to encounter for open-weight language models. Without GGUF
support, Zigrad would need bespoke conversion steps before it can participate in
that ecosystem.

## Goals

- Parse GGUF metadata and tensor tables.
- Load common floating-point and quantized tensor formats.
- Map GGUF tensor layouts into Zigrad tensor representations.
- Enable LLM reference examples to consume external checkpoints.
- Expose a clear unsupported-format policy and diagnostics.

## Non-Goals

- Full llama.cpp runtime parity.
- Supporting every historical ggml variant immediately.
- Implementing all quantized kernels in the first milestone.
- Shipping tokenizer and prompt-engineering infrastructure in this RFC.

## Scope

Initial scope includes:

- GGUF file parsing,
- metadata extraction,
- tensor name and shape mapping,
- memory-mapped or streaming load strategies,
- support for common unquantized and priority quantized formats,
- integration hooks for example models.

## Architecture

Introduce `src/interop/gguf/` with:

- container parser,
- metadata schema types,
- tensor descriptor table,
- load planner,
- tensor conversion and dequantization utilities,
- diagnostics and compatibility reporting.

The loader must distinguish:

- zero-copy mapping opportunities,
- eager decode paths,
- dequantize-on-load paths,
- future dequantize-on-demand paths.

## Format Support Strategy

The first milestone should prioritize:

- unquantized weights,
- a small set of widely used quantized layouts,
- model families needed by the reference examples.

Every supported GGUF tensor type must declare:

- whether it loads losslessly,
- whether it executes natively,
- whether it requires dequantization,
- whether export is supported.

## Execution Strategy

The loader should support two modes:

- `materialize`: convert tensors into Zigrad-owned tensors on load,
- `backed`: retain memory-mapped storage where safe and beneficial.

Quantized execution may start with dequantize-on-load for correctness and
simpler integration. Native quantized kernels can arrive later.

## Example Integration Requirements

Reference LLM examples must be able to:

- load checkpoint metadata,
- validate expected tensor names and shapes,
- report missing or remapped tensors,
- optionally load only a model subset for debugging.

## Testing Plan

- parser tests on curated GGUF fixtures,
- tensor mapping tests for representative model shards,
- load tests for memory usage and partial loading,
- execution parity tests on a reference LLM block where feasible,
- benchmark coverage for checkpoint load and first-token latency.

## Milestones

### Milestone A: Reader

- GGUF metadata parser,
- tensor table decoding,
- basic floating-point tensor load.

### Milestone B: Quantized Compatibility

- support priority quantized formats,
- dequantize-on-load path,
- diagnostics for unsupported layouts.

### Milestone C: Example Enablement

- LLM example consumes GGUF weights,
- benchmark coverage for load and inference startup.

## Acceptance Criteria

- At least one GGUF-backed reference model loads into Zigrad.
- Loader diagnostics clearly identify unsupported tensors or metadata.
- Memory behavior of the loader is benchmarked and documented.
- Reference example can run at least a minimal inference path.

## Risks

- Quantized layouts can require backend-specific kernel work quickly.
- Model naming and tensor layout conventions may vary across ecosystems.
- Memory mapping semantics differ across operating systems.

## Open Questions

- Do we want export to GGUF, or only import, in the first iteration?
- Which quantized formats are the true must-have subset?
- Should quantized tensors remain first-class or be normalized eagerly?

