# Next Milestones Specification

Last updated: `2026-03-30`

This document specifies the next high-priority milestones for Zigrad in
implementation-ready detail. Each milestone is scoped to a single coherent
deliverable with explicit acceptance criteria, task breakdown, and dependency
mapping.

Milestones are ordered by priority. An agent should start at M-1 and work
downward, skipping only if blocked.

---

## Current State Summary

The compiler stack has reached the following state:

| Layer | State | Key Files |
|-------|-------|-----------|
| Eager execution | Stable, production-quality | `src/ndtensor.zig`, `src/ndarray.zig` |
| Lazy capture (observe) | Landed, tested | `src/lazy.zig` |
| Lazy capture (deferred) | Landed, forward-only | `src/lazy.zig`, `src/device/device_reference.zig` |
| Graph IR | Landed, SSA-form with verifier | `src/graph_ir.zig` |
| Pass manager | Landed, timing + verification | `src/graph_ir.zig` |
| DCE pass | Landed, working | `src/graph_ir.zig` |
| Constant folding | **Landed** | `src/graph_ir.zig` |
| Algebraic simplification | **Landed** | `src/graph_ir.zig` |
| Execution bridge | **Landed** | `src/graph_ir.zig` |
| ONNX interop | Not started | — |
| GGUF interop | Not started | — |

---

## M-1: Graph IR Execution Bridge ✓ LANDED

**RFC:** RFC-0007 (Static Graph Optimization)
**Priority:** Highest — unblocks constant folding, algebraic simplification, and
end-to-end optimized execution.
**Depends on:** Graph IR (landed), deferred execution (landed)
**Blocks:** M-2, M-3, M-5
**Status:** Complete. Landed 2026-03-30.

### Goal

Lower an optimized `GraphIR` back to concrete device execution so that a
captured-and-optimized graph produces the same results as eager execution,
measurably.

### Scope

1. **IR interpreter**: Walk the `GraphIR` ops in topological order and dispatch
   each op through `DeviceReference.dispatch()` using the operand/result value
   buffers.
2. **Buffer planner**: Allocate device buffers for each IR value, reusing buffers
   for values whose lifetime has ended (dead after last consumer).
3. **`GraphIR.execute()`** public method that:
   - Allocates buffers for all values.
   - Binds graph-input buffers to user-provided input tensors.
   - Walks ops in topo order, dispatching each.
   - Returns output buffers mapped to graph-output value IDs.
4. **Roundtrip parity tests**: Eager execution vs. capture-optimize-execute
   produces identical results on representative workloads (elwise chain, matmul
   chain, MNIST-like forward pass).

### Non-scope

- JIT compilation or code generation (RFC-0008).
- Backward pass through the IR interpreter.
- Automatic fallback from IR execution to eager on unsupported ops.

### Deliverables

| # | Deliverable | File(s) | Acceptance |
|---|-------------|---------|------------|
| 1 | Topological sort utility for GraphIR | `src/graph_ir.zig` | Returns ops in valid execution order; verified by verifier |
| 2 | Buffer allocation plan | `src/graph_ir.zig` | Maps value IDs to allocated device buffers; lifetime-based reuse |
| 3 | Op dispatch mapping | `src/graph_ir.zig` | Maps IR op names to `opspec` dispatch calls; covers elwise ops, transpose, matmul/bmm, relu, softmax |
| 4 | `GraphIR.execute()` method | `src/graph_ir.zig` | Takes input tensors, returns output tensors; verified against eager |
| 5 | Roundtrip parity tests | `src/graph_ir.zig` | At least 3 tests: elwise, matmul, multi-layer forward |
| 6 | Benchmark comparison | `src/graph_ir.zig` or benchmarks | Capture vs eager overhead measured |

### Task Breakdown

```
1. Add topological sort to GraphIR (sort ops by dependency order)
2. Design buffer allocation plan struct (value_id -> DeviceData mapping)
3. Implement op-name-to-opspec dispatch table for core ops:
   - ADD, SUB, MUL, DIV (elwise)
   - MATMUL, MATMUL_ABt, bmm_acc (BLAS)
   - TRANSPOSE
   - relu_fwd, exp_fwd, sqrt_fwd, tanh_fwd (activations)
   - softmax, sum, max_along (reductions)
4. Implement GraphIR.execute() that:
   a. Topo-sorts ops
   b. Allocates output buffers
   c. Binds input buffers from user-provided tensors
   d. Dispatches each op using the dispatch table
   e. Returns outputs
5. Write roundtrip parity tests
6. Add benchmark comparison (optional, time permitting)
```

### Key Design Decision

The dispatch table maps `op.name` (a string like `"ADD"`) to the appropriate
`opspec` constructor and field mapping. This is a comptime switch or a runtime
string-match dispatch. Since IR op names come from lazy capture (which uses
`@tagName(op)` or explicit strings), the mapping is stable.

The challenge is that `opspec` types are parameterized by element type `T`,
but the IR `DType` is a runtime enum. The dispatch table must do a runtime
dtype switch:

```zig
switch (value.dtype) {
    .f32 => dispatchOp(f32, op, buffers, device),
    .f64 => dispatchOp(f64, op, buffers, device),
    else => return error.UnsupportedDType,
}
```

---

## M-2: Constant Folding Pass ✓ LANDED

**RFC:** RFC-0007 (Static Graph Optimization)
**Priority:** High — first real optimization that changes execution behavior.
**Depends on:** M-1 (execution bridge)
**Blocks:** M-3
**Status:** Complete. Landed 2026-03-30.

### Goal

Evaluate ops whose inputs are all compile-time constants (graph inputs with
known values) and replace them with their computed results in the IR.

### Scope

1. **Constant tracking**: Extend `Value` with an optional `constant_data: ?[]const u8`
   field that stores the raw bytes of known-constant values.
2. **`GraphIR.fromSession()` enhancement**: When lowering source tensors that
   have stable data (e.g., weight matrices), optionally capture their data as
   constants.
3. **Fold logic**: For each op where all operands have `constant_data`, execute
   the op using the execution bridge, capture the result, replace the op's
   result value with a constant, and mark the op as dead.
4. **DCE cleanup**: Run DCE after folding to remove the now-dead ops.

### Deliverables

| # | Deliverable | Acceptance |
|---|-------------|------------|
| 1 | `Value.constant_data` field | Optional raw bytes storage |
| 2 | Constant propagation in `fromSession()` | Source tensors carry data |
| 3 | `constantFoldPass()` implementation | Folds ops with all-constant inputs |
| 4 | Tests | At least 2: fold scalar ops, fold with partial constants |

### Task Breakdown

```
1. Add constant_data: ?[]const u8 to Value struct
2. In fromSession(), for "source" tensors, optionally capture raw data
3. Implement fold logic:
   a. Identify foldable ops (all operands have constant_data)
   b. Use execution bridge to evaluate the op
   c. Store result as constant_data on the result value
   d. Mark op for removal
4. Wire into the existing constantFoldPass() stub
5. Run DCE after folding
6. Add tests: constant arithmetic, mixed constant/variable
```

---

## M-3: Algebraic Simplification Pass ✓ LANDED

**RFC:** RFC-0007 (Static Graph Optimization)
**Priority:** High — cheap optimization with broad applicability.
**Depends on:** M-2 (constant tracking on values)
**Blocks:** None directly
**Status:** Complete. Landed 2026-03-30.

### Goal

Apply identity and annihilator rules to simplify the graph:
- `x + 0 -> x`, `x - 0 -> x`
- `x * 1 -> x`, `x / 1 -> x`
- `x * 0 -> 0`
- `x - x -> 0`

### Scope

1. **Zero/one detection**: Check if a value's `constant_data` represents all-zeros
   or all-ones for its dtype.
2. **Rewrite rules**: For each supported pattern, replace the op's result with
   the appropriate operand (identity) or a new zero constant (annihilator).
3. **Use-def update**: Update all consumers of the replaced value to reference
   the simplified value.

### Deliverables

| # | Deliverable | Acceptance |
|---|-------------|------------|
| 1 | Zero/one constant detection helpers | Works for f32, f64 |
| 2 | Identity rewrite rules | x+0, x*1, x/1 eliminated |
| 3 | Annihilator rewrite rules | x*0 folded to zero constant |
| 4 | Tests | At least 3 distinct simplification patterns verified |

### Task Breakdown

```
1. Add isZeroConstant(value) and isOneConstant(value) helpers
2. Implement identity rules: scan ops, detect identity operands, rewrite
3. Implement annihilator rules: detect zero multiplier, create zero result
4. Handle use-def rewiring when replacing a value
5. Wire into the existing algebraicSimplifyPass() stub
6. Add tests
```

---

## M-4: ONNX Import MVP

**RFC:** RFC-0004 (ONNX Interop)
**Priority:** High — first external model format, critical for ecosystem.
**Depends on:** RFC-0007 Graph IR (landed)
**Blocks:** None immediately

### Goal

Parse an ONNX `.onnx` protobuf model file and lower it to a Zigrad `GraphIR`
that can be verified and (with M-1) executed.

### Scope

1. **ONNX protobuf parser**: Minimal protobuf wire-format reader for ONNX's
   `ModelProto` / `GraphProto` / `NodeProto` / `TensorProto` schemas. No
   codegen dependency — hand-rolled reader targeting the specific ONNX message
   structure.
2. **Op registry**: Map ONNX op names (`Add`, `MatMul`, `Relu`, `Softmax`,
   `Transpose`, `Reshape`, `Conv`) to Zigrad IR op names and attribute
   translations.
3. **Tensor import**: Load ONNX `TensorProto` initializers as constant values
   in the IR.
4. **`onnx.importModel(allocator, bytes) !GraphIR`**: Top-level import function.
5. **Validation**: Imported IR passes the verifier.

### Non-scope

- ONNX export (Milestone M-4b, later).
- Training graph support.
- Full ONNX opset coverage (start with opset 13+ core ops).

### Deliverables

| # | Deliverable | File(s) | Acceptance |
|---|-------------|---------|------------|
| 1 | Protobuf wire reader | `src/interop/onnx/proto.zig` | Parses varint, length-delimited, fixed fields |
| 2 | ONNX schema types | `src/interop/onnx/schema.zig` | ModelProto, GraphProto, NodeProto, TensorProto |
| 3 | Op registry | `src/interop/onnx/ops.zig` | Maps ONNX op names to IR ops |
| 4 | Import function | `src/interop/onnx/import.zig` | `importModel(allocator, bytes) !GraphIR` |
| 5 | Tests | `src/interop/onnx/` | Import a small test model, verify IR structure |
| 6 | Docs | `docs/rfcs/0004-onnx-interop.md` | Agentic context updated |

### Task Breakdown

```
1. Create src/interop/onnx/ directory structure
2. Implement minimal protobuf wire-format reader (varint, LEB128, zigzag,
   length-delimited, fixed32/64)
3. Define ONNX schema types (ModelProto, GraphProto, NodeProto, TensorProto,
   AttributeProto, ValueInfoProto)
4. Implement protobuf-to-schema parsing
5. Build op registry mapping ONNX op names to Zigrad IR op names + attribute
   translation (start with: Add, Sub, Mul, Div, MatMul, Relu, Sigmoid, Tanh,
   Softmax, Transpose, Reshape, Gemm)
6. Implement TensorProto -> constant Value lowering (f32, f64, int types)
7. Implement GraphProto -> GraphIR lowering (inputs, outputs, initializers, nodes)
8. Implement importModel() top-level function
9. Create a small test ONNX model (2-layer MLP) as test fixture
10. Write import + verify tests
```

---

## M-5: GGUF Reader MVP

**RFC:** RFC-0005 (ggml/GGUF Interop)
**Priority:** High — enables LLM weight loading.
**Depends on:** RFC-0012 (landed)
**Blocks:** None immediately

### Goal

Parse a GGUF container file and load tensor weights into Zigrad `NDArray`
buffers, enabling LLM example models to load pre-trained weights.

### Scope

1. **GGUF container parser**: Read the GGUF header, metadata key-value pairs,
   and tensor descriptor table from a binary file.
2. **Tensor loading**: Map GGUF tensor descriptors to `NDArray` allocations,
   supporting at minimum `f32` and `f16` data types.
3. **Dequantization**: Support at least Q4_0 and Q8_0 quantized formats with
   dequantize-on-load to f32.
4. **`gguf.loadTensors(allocator, path, device) !TensorMap`**: Top-level load
   function returning a name-to-NDArray map.

### Non-scope

- GGUF export/write.
- Full quantization format coverage (start with Q4_0, Q8_0, f16, f32).
- Quantized-tensor arithmetic (always dequantize to f32 on load).

### Deliverables

| # | Deliverable | File(s) | Acceptance |
|---|-------------|---------|------------|
| 1 | GGUF header parser | `src/interop/gguf/parser.zig` | Reads magic, version, tensor count, metadata count |
| 2 | Metadata reader | `src/interop/gguf/parser.zig` | Reads all GGUF metadata types |
| 3 | Tensor descriptor table | `src/interop/gguf/parser.zig` | Name, shape, dtype, offset for each tensor |
| 4 | f32/f16 tensor loader | `src/interop/gguf/loader.zig` | Loads into NDArray with correct shape |
| 5 | Q4_0/Q8_0 dequantizer | `src/interop/gguf/quant.zig` | Dequantize to f32 on load |
| 6 | Top-level API | `src/interop/gguf/root.zig` | `loadTensors()` function |
| 7 | Tests | `src/interop/gguf/` | Round-trip: write known data in GGUF format, load back, verify |

### Task Breakdown

```
1. Create src/interop/gguf/ directory structure
2. Implement GGUF header parsing (magic bytes, version, counts, alignment)
3. Implement metadata key-value reader (all GGUF value types: u8..u64,
   f32, f64, bool, string, array)
4. Implement tensor descriptor table reader (name, ndims, shape, dtype, offset)
5. Implement f32 tensor loading (mmap or read + copy to device)
6. Implement f16 tensor loading with f16-to-f32 conversion
7. Implement Q4_0 dequantization (block size 32, 4-bit quantized + f16 scale)
8. Implement Q8_0 dequantization (block size 32, 8-bit quantized + f16 scale)
9. Implement loadTensors() top-level function
10. Create a small test GGUF fixture (generate via script or embed binary)
11. Write load + verify tests
12. Export module through zigrad.zig
```

---

## M-6: Deferred Backward Pass

**RFC:** RFC-0006 (Lazy Tensors)
**Priority:** Medium — completes the lazy tensor story for training workloads.
**Depends on:** Deferred execution (landed)
**Blocks:** RFC-0008 (full dynamic compiler)

### Goal

Extend deferred execution mode to support `backward()` calls, so a full
training step (forward + loss + backward + optimizer) can be captured and
deferred.

### Scope

1. **Backward thunk capture**: When `backward()` is called in deferred mode,
   the backward graph traversal should also enqueue thunks rather than
   executing immediately.
2. **Gradient buffer allocation**: Pre-allocate gradient buffers at capture
   time so backward thunks have valid output targets.
3. **Realize semantics**: `realize()` on a loss tensor flushes both forward
   and backward thunks, producing gradients.

### Non-scope

- Optimized backward scheduling (simple FIFO for now).
- Higher-order gradients in deferred mode.

### Deliverables

| # | Deliverable | Acceptance |
|---|-------------|------------|
| 1 | Backward ops enqueue thunks in deferred mode | Backward dispatch deferred |
| 2 | Gradient buffer pre-allocation | Gradient buffers exist before backward flush |
| 3 | Forward+backward roundtrip test | Deferred gradients match eager gradients |
| 4 | Training step test | Forward + loss + backward in deferred mode |

### Task Breakdown

```
1. Analyze how backward() dispatches ops (trace through graph.zig backward)
2. Identify all backward dispatch paths that need thunk interception
3. Implement gradient buffer pre-allocation for deferred mode
4. Ensure backward ops go through DeviceReference.dispatch() (already true
   for most — verify)
5. Add realize() semantics for backward (flush forward thunks, then backward)
6. Write parity tests: eager backward vs deferred backward
7. Write training step test: forward + loss + backward + check gradients
```

---

## M-7: Common Subexpression Elimination (CSE) Pass

**RFC:** RFC-0007 (Static Graph Optimization)
**Priority:** Medium — reduces redundant computation in captured graphs.
**Depends on:** Graph IR (landed)
**Blocks:** None

### Goal

Identify and merge duplicate operations in the IR — ops with the same name,
same operands, and same attributes produce the same result and can be shared.

### Scope

1. **Op identity hashing**: Hash an op by (name, operand IDs, attributes).
2. **Dedup scan**: For each op, check if an identical op already exists.
   If so, rewrite all uses of the duplicate's results to reference the
   original's results, and mark the duplicate dead.
3. **DCE cleanup**: Run DCE after CSE.

### Deliverables

| # | Deliverable | Acceptance |
|---|-------------|------------|
| 1 | Op identity hash function | Deterministic hash of name+operands+attrs |
| 2 | CSE pass | Merges duplicate ops, rewrites uses |
| 3 | Tests | Duplicate add eliminated, non-duplicate preserved |

---

## Milestone Dependency Graph

```
M-1 Execution Bridge
 |
 +---> M-2 Constant Folding
 |      |
 |      +---> M-3 Algebraic Simplification
 |
 +---> M-4 ONNX Import (also needs M-1 for execution)

M-5 GGUF Reader (independent, can run in parallel with M-1..M-3)

M-6 Deferred Backward (independent of M-1..M-5)

M-7 CSE Pass (independent, needs only existing Graph IR)
```

**Recommended parallelization:**
- Start M-1 (execution bridge) as the critical path
- M-5 (GGUF reader) and M-7 (CSE pass) can run in parallel with M-1
- M-4 (ONNX import) can start parsing work in parallel, but execution
  testing needs M-1
- M-2 and M-3 are sequential after M-1
- M-6 is independent and can be scheduled based on training-workload demand

---

## Success Criteria for the Next Phase

Milestones M-1 through M-3 are now complete (landed 2026-03-30):
- A user can capture a forward pass in deferred mode, lower to IR, run DCE +
  constant folding + algebraic simplification, and execute the optimized graph
  with identical results to eager mode. ✓ Verified via roundtrip parity tests.
- Constant subexpressions are evaluated at optimization time and replaced
  with their computed values. ✓ Verified.
- Identity (x+0, x*1, x/1) and annihilator (x*0) patterns are eliminated
  from the graph. ✓ Verified.

When M-4 and M-5 are also complete:
- A user can load an ONNX model or GGUF weights into Zigrad and execute
  inference, bridging the gap between external model formats and the Zigrad
  runtime.
