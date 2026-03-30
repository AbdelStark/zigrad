# RFC-0005: ggml and GGUF Interoperability

Status: `In Progress` (Milestone A complete)
Priority: `P1`
Depends on: RFC-0001, RFC-0012
Blocks: None
Last updated: `2026-03-30`

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

## Agentic Context

### 2026-03-30 Dependency Readiness

- Dependencies now available:
  - **RFC-0001 (Benchmarking):** Fully landed. Interop benchmark contract
    ready for GGUF load benchmarks once the artifact path exists.
  - **RFC-0012 (Examples):** All reference examples landed (MNIST, char-LM,
    pendulum, corridor, DQN, GCN). The char-LM example is the most natural
    consumer for GGUF-loaded weights.
- This RFC is now **unblocked** for implementation.
- Recommended approach: GGUF loads to `NDArray` buffers (not `GraphIR`),
  since weights are data, not computation. The loaded tensors can then be
  used as inputs to either eager execution or lazy-captured graphs.
- Detailed milestone spec: see [`docs/next-milestones.md`](../next-milestones.md) M-5.

### 2026-03-30 Milestone A: Reader — Complete

- **Landed:** Full GGUF reader MVP in `src/interop/gguf/`:
  - `parser.zig`: GGUF container parser — header, metadata KV pairs (all 13
    value types including nested arrays), tensor descriptor table. Supports
    GGUF v2 and v3, configurable alignment.
  - `quant.zig`: Dequantize-on-load for f32 (passthrough), f16→f32, Q4_0→f32,
    Q8_0→f32. Block-level dequantization matching the ggml reference.
  - `loader.zig`: Top-level `loadTensors(allocator, data, device, options)`
    returns a `TensorMap` (name→`NDArray(f32)` + original dtype). Supports
    partial loading via name filter. Skips unsupported tensor types with
    diagnostic log.
  - `root.zig`: Module re-exports following the ONNX interop pattern.
- **Exported:** `pub const gguf = @import("interop/gguf/root.zig")` in
  `src/zigrad.zig`. Accessible as `zg.gguf.loadTensors(...)`.
- **Tests:** 17 inline tests covering:
  - Reader primitives, string parsing, alignment
  - Full GGUF file parse with metadata and tensors
  - All metadata value types including arrays
  - Invalid magic and unsupported version rejection
  - Multi-tensor files with mixed dtypes
  - f32/f16/Q4_0/Q8_0 dequantization roundtrips
  - End-to-end tensor loading onto host device
  - Partial loading with name filter
  - Metadata access through TensorMap
- **Validation:** `zig build test` passes with 0 compilation errors.
- **What remains:**
  - Milestone B: Additional quantized formats (Q4_1, Q5_0, Q5_1, Q2_K..Q8_K)
  - Milestone C: Example integration (LLM example consuming GGUF weights)
  - GGUF export (non-goal for initial scope)
  - Memory-mapped loading for large files
  - Benchmark coverage for load throughput
