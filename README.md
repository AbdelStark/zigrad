<p align="center">
  <img src="./docs/zg-logo.svg" width=350>
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/AbdelStark/zigrad?style=flat&logo=opensourceinitiative" alt="license">
	<img src="https://img.shields.io/github/last-commit/AbdelStark/zigrad?style=flat&logo=git&logoColor=white" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/AbdelStark/zigrad?style=flat&color=F7A41D" alt="repo-top-language">
</p>
<br>
<p align="center" class="markdown-heading"><strong><i>Supporting AI innovation from ideation to results.</i></strong></p>

---

> **Note on project lineage.** This repository is based on the original
> [Zigrad](https://github.com/Marco-Christiani/zigrad) project created by
> Marco Christiani. It is not an official fork — it is an independent
> continuation that has undergone substantial restructuring, refactoring,
> and new development. The original design philosophy and core ideas are
> credited to the upstream project. Given the scope of changes already
> landed and the direction of ongoing work, this will likely evolve into a
> standalone project. Contributions and ideas from the original remain
> acknowledged and appreciated.

---

> **This project is under heavy active development.**
> APIs are changing rapidly, new subsystems are being built, and the
> codebase is being restructured. Bug reports, benchmarks, and
> contributions are welcome.

> **CUDA support is experimental.**
> CUDA integration is in beta and may be incomplete or unstable. Full GPU
> support is actively being developed.

---

## What is Zigrad?

Zigrad is a machine learning framework written in Zig, designed to bridge
the gap between research iteration speed and production training
infrastructure. Instead of forcing a choice between a high-level research
framework and a high-performance training system, Zigrad provides a single
stack where you can:

- Experiment using high-level, PyTorch-like abstractions
- Gradually opt into fine-grained control and performance optimizations
- Access low-level primitives without switching frameworks or writing C extensions
- Load models from standard formats (ONNX, GGUF, SafeTensors)
- Capture, optimize, and replay computation graphs
- Run on constrained hardware where Python runtimes are impractical

## Performance

2.5x+ speedup over a compiled PyTorch model on Apple Silicon, 1.5x on x86.

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/zg_mnist_zg_torch_perf.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/zg_mnist_zg_torch_perf_dark.svg" >
  <img alt="Zigrad vs PyTorch benchmark comparison" src="docs/zg_mnist_zg_torch_perf.svg">
</picture>

<sub>*Tensorflow excluded for scaling purposes (too slow).</sub>

## Current Capabilities

### Automatic Differentiation

General-purpose autograd engine for n-dimensional data with eager execution
and dynamic computation graphs. Supports custom differentiable operations and
computation graph tracing/visualization.

### Lazy Tensors and Deferred Execution

Opt-in lazy evaluation via `zg.lazy.Session`:

- **Observe mode**: Operations execute eagerly while the session records a
  stable tensor graph for inspection or lowering to IR.
- **Deferred mode**: Operations queue as thunks and replay on `realize()`.
  Both forward and backward passes can be deferred, enabling full training
  step capture.

```zig
var session = zg.lazy.Session.init(allocator);
session.mode = .deferred;

var capture = try session.begin();
defer capture.end();

const y = try x.add(w);
const z = try y.mul(x);
_ = try z.realize();  // flushes all pending work
```

### Graph IR and Optimization Pipeline

Captured computation graphs can be lowered to a typed SSA-form Graph IR and
optimized through a composable pass pipeline:

- **Dead Code Elimination** — removes unused ops
- **Constant Folding** — evaluates compile-time-known subexpressions
- **Algebraic Simplification** — eliminates identity/annihilator patterns
  (x+0, x\*1, x\*0, x/1)
- **Common Subexpression Elimination** — merges duplicate operations

All passes are orchestrated via `PassManager` with per-pass timing and
optional pre/post verification.

### Model Format Interop

**ONNX Import** — Parse `.onnx` protobuf models and lower to Zigrad GraphIR.
Covers core ops (Add, Sub, Mul, Div, MatMul, Gemm, Relu, Sigmoid, Tanh,
Softmax, Transpose, and more).

```zig
const ir = try zg.onnx.importModel(allocator, onnx_bytes);
```

**GGUF Reader** — Load GGUF model weights (the standard format for
llama.cpp and open-weight LLMs) into Zigrad tensors. Supports f32, f16,
Q4_0, and Q8_0 with dequantize-on-load to f32.

```zig
var tensors = try zg.gguf.loadTensors(allocator, gguf_bytes, device, .{});
const weights = tensors.get("model.layers.0.attention.wq.weight");
```

**SafeTensors** — Pure-Zig SafeTensors implementation for checkpoint
save/load across all standard dtypes.

### Backend Support

| Backend | Build Flag | Status |
|---------|-----------|--------|
| Apple Accelerate | default on macOS | Stable |
| OpenBLAS | `-Dhost_blas=openblas` | Stable |
| Intel MKL | `-Dhost_blas=mkl` | Stable |
| CUDA | `-Denable_cuda=true` | Experimental |

Runtime backend selection via `ZG_DEVICE=host|cpu|cuda[:index]`.

### Neural Network Primitives

- **Layers**: Linear, Conv2D (im2col), ReLU, Tanh, Sigmoid
- **Loss**: MSE, Softmax Cross-Entropy, NLL, Smooth L1
- **Optimizers**: SGD, Adam (with device-dispatched updates)
- **Integrations**: TensorBoard scalar/histogram logging, MuJoCo (via examples)

### CommitLLM — Verifiable INT8 Inference

Zig port of the [CommitLLM](https://github.com/lambdaclass/CommitLLM)
cryptographic commit-and-audit protocol for open-weight LLM inference.
Based on the [CommitLLM paper](https://github.com/lambdaclass/CommitLLM/blob/main/paper/main.pdf)
and [Rust reference implementation](https://github.com/lambdaclass/CommitLLM/tree/main/crates/verilm-core).

The provider serves on GPU with ~12% tracing overhead and returns a compact
receipt. An auditor verifies on CPU in ~1.3ms per challenged token (Llama 70B)
— without re-running inference.

**Core primitives:**

| Component | Description |
|-----------|-------------|
| **Freivalds checks** | Precompute `v = r^T W`, verify `v·x == r·z` in O(n) per matrix. False-accept ≤ 1/p. |
| **Three field sizes** | Fp (p=2³²-5), Fp64 (p=2⁶¹-1), Fp128 (p=2¹²⁷-1) — Mersenne primes for fast reduction. |
| **SHA-256 Merkle trees** | Domain-separated trace commitments with splice/reorder-resistant IO chains. |
| **Q8_0 block Freivalds** | Batched verification for GGML-style quantized weight blocks. |
| **SiLU verification** | LUT-based (toy) and scaled f64 (production W8A8) paths. |

**Run the E2E showcase** (8-phase annotated demo):

```shell
make commitllm-e2e-showcase
# or: zig build commitllm-showcase
```

Output walks through model setup → keygen → forward pass → 14 Freivalds checks
→ Merkle commitment → tamper detection → field arithmetic → logit binding, all
with intermediate values and timing.

**Run the cross-language showcase** (Rust prover → Zig verifier):

```shell
make commitllm-e2e-crosslang-showcase
```

The Rust prover generates a complete proof bundle (weights, traces, Merkle
commitment, IO chain). The Zig verifier independently recomputes every
check — then runs adversarial tamper detection:

| Phase | What | Result |
|-------|------|--------|
| 1–6 | Honest verification: v precompute, Freivalds, Merkle, weight hash, IO chain | All match cross-language |
| 7 | Flip one i32 matmul accumulator → Freivalds detects | `v·x ≠ r·z'` (false-accept 2.3e-10) |
| 8 | Flip one byte in retained state → Merkle root diverges | Root mismatch detected |
| 9 | Wrong token ID, broken chain link, swapped leaf, weight substitution | All 4 attacks caught |

**Run all commitllm tests** (73 tests, including differential tests against Rust):

```shell
make commitllm-test
```

**Module location:** `src/commitllm/` — 11 files, ~3500 lines.

### Benchmark Harness

A comprehensive benchmarking program with:
- Machine-readable JSONL output with hardware and backend metadata
- Regression detection (warn >5%, fail >10%)
- PyTorch baseline comparison
- Provider-specific reports (Accelerate vs OpenBLAS vs MKL)
- Thread-scaling analysis
- Smoke suites for CI

```shell
zig build benchmark
zig build benchmark -- --baseline pytorch
zig build benchmark-compare -- --baseline results/baseline.jsonl --candidate results/latest.jsonl
```

### Computation Graph Visualization

![](./docs/comp_graph_mnist_simple_noag.svg)

Trace and render the computation graph generated by any model. Works with
both the module API and raw autograd operations.

## Getting Started

### Prerequisites

Only dependency is a BLAS library.

### Apple Silicon (macOS)

Uses Accelerate by default:

```shell
git clone https://github.com/AbdelStark/zigrad/
cd zigrad
zig build test
```

### Linux

- **OpenBLAS**: `zig build -Dhost_blas=openblas`
- **MKL** (recommended for best performance):
  `zig build -Dhost_blas=mkl -Dmkl_include_dir=/path/to/include -Dmkl_library_dir=/path/to/lib`

### Examples

Seven reference examples, all runnable from a clean checkout:

```shell
# Hello world — minimal autograd demo
cd examples/hello-world && zig build run

# MNIST MLP
cd examples/mnist && make

# Satoshi-style causal self-attention language model
cd examples/satoshi-lm && zig build run

# Pendulum dynamics regression
cd examples/pendulum && zig build run

# Deterministic corridor-control RL
cd examples/corridor && zig build run

# CartPole DQN with TensorBoard logging
cd examples/dqn && zig build run

# Two-layer Graph Convolutional Network
cd examples/gcn && zig build run

# CommitLLM E2E showcase — verifiable inference demo
zig build commitllm-showcase

# CommitLLM cross-language — Rust prover, Zig verifier + tamper detection
make commitllm-e2e-crosslang-showcase
```

All examples support smoke mode for fast validation:

```shell
ZG_EXAMPLE_SMOKE=1 zig build run
```

CUDA-enabled examples support runtime backend selection:

```shell
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```

## Architecture

```
src/
  zigrad.zig          # Public API surface
  ndarray.zig         # N-dimensional array (backend-agnostic)
  ndtensor.zig        # Autograd tensor with gradient tracking
  graph.zig           # Dynamic computation graph and backward pass
  graph_ir.zig        # Static SSA-form IR, verifier, optimization passes
  lazy.zig            # Lazy session capture and deferred execution
  nn/                 # Neural network layers, loss functions, optimizers
  device/             # Backend abstraction (host, CUDA)
  commitllm/          # CommitLLM verifiable inference (Freivalds, Merkle, fields)
  interop/
    onnx/             # ONNX protobuf parser and GraphIR importer
    gguf/             # GGUF container parser and tensor loader
  third_party/
    safetensors_zg/   # Pure-Zig SafeTensors implementation
tensorboard/          # TensorBoard event file writer
benchmarks/           # Benchmark harness, specs, and comparison tools
examples/             # Reference models and getting-started demos
docs/
  rfcs/               # Design documents for each major subsystem
  roadmap.md          # Implementation roadmap and RFC index
```

## Roadmap

The detailed implementation plan lives in [`docs/roadmap.md`](./docs/roadmap.md),
with per-subsystem RFCs in [`docs/rfcs/`](./docs/rfcs/).

**Completed:**
- Standardized benchmark harness with regression detection
- Multi-provider host backend (Accelerate, OpenBLAS, MKL)
- CUDA backend (experimental, pending GPU validation)
- Lazy tensors with deferred forward and backward execution
- Static graph optimization (DCE, constant folding, algebraic simplification, CSE)
- ONNX import MVP
- GGUF reader MVP (f32, f16, Q4_0, Q8_0)
- Seven reference examples (MNIST, char-LM, pendulum, corridor, DQN, GCN)
- CommitLLM verifiable inference module (Freivalds, Merkle, Fp/Fp64/Fp128, Q8_0)

**In progress / planned:**
- Dynamic graph compiler
- ONNX export
- Additional GGUF quantized formats
- MLIR lowering pipeline
- LLM reference examples consuming GGUF weights
- ZML inference bridge
- Apache TVM integration

## Known Issues and Limitations

- Documentation is sparse as the API is still stabilizing. The examples are
  the best quickstart guides.
- Conv and pooling layers are test implementations — functional but unoptimized.
- CUDA support requires real GPU hardware validation (pending).

## Contributing

- Any open issue is available for development
- Please open an issue first before working on a PR
- Bug reports and benchmark contributions are especially welcome

## License

See [LICENSE](./LICENSE).
