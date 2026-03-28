<p align="center">
  <img src="./docs/zg-logo.svg" width=350>
</p>

<p align="center">
	<img src="https://img.shields.io/github/license/Marco-Christiani/zigrad?style=flat&logo=opensourceinitiative" alt="license">
	<img src="https://img.shields.io/github/last-commit/Marco-Christiani/zigrad?style=flat&logo=git&logoColor=white" alt="last-commit">
	<img src="https://img.shields.io/github/languages/count/Marco-Christiani/zigrad?style=flat" alt="repo-language-count">
	<img src="https://img.shields.io/github/languages/top/Marco-Christiani/zigrad?style=flat&color=F7A41D" alt="repo-top-language">
	<!-- <img src="https://img.shields.io/badge/Zig-F7A41D.svg?style=flat&logo=Zig&logoColor=white" alt="Zig"> -->
	<!-- 1325584101809324113 -->
	<img alt="Discord" src="https://img.shields.io/discord/1325584101809324113?style=flat">
</p>
<br>
<p align="center" class="markdown-heading"><strong><i>Supporting AI innovation from ideation to results.</i></strong></p>

---
> ⚠️ **Zigrad is undergoing a rewrite**
> Public release tentatively planned for mid 2026 please stay tuned.

> 🚧 **Zigrad is under active development.**  
> By using Zigrad, you are participating in its development and contributing to its early testing and validation. Expect APIs to change and features to evolve rapidly. Bug reports, benchmarks, and contributions are welcome.

> 🧪 **CUDA support is experimental.**  
> CUDA integration is in **beta** and may be incomplete, unstable, or suboptimal. Use it for testing and feedback. Full GPU support is actively being developed.
---

AI frameworks optimized for rapid research iteration do not seamlessly transition into the infrastructure required for large-scale training. This fragmented pipeline creates redundant engineering effort and slows iteration cycles. Zigrad provides a path to performance that preserves the natural development workflow researchers prefer; bridging research and engineering. Using Zigrad you can:

  - Experiment using high-level, PyTorch-like abstractions
  - Gradually opt into fine-grained control and performance optimizations
  - Access low-level primitives and assert control--without switching frameworks, code translation, or building complex extensions.
  - Quickly transition research to high performance training
  - Learn on the edge with SWAP constrained hardware


https://github.com/user-attachments/assets/3842aa72-9b16-4c25-8789-eac7159e3768

**Fast**
<!-- benchmarks -->

2.5x+ speedup over a compiled PyTorch model on Apple Silicon, 1.5x on x86. Expect similar performance gains across more architectures and platforms as MKL/CUDA support improves and Zigrad's ML graph compiler is operational.
<!-- link to a benchmarking page -->
<!-- only need one of the bm plots, probably fast vs fast since that requires the least explanation -->

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/zg_mnist_zg_torch_perf.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/zg_mnist_zg_torch_perf_dark.svg" >
  <img alt="Description of the image" src="docs/zg_mnist_zg_torch_perf.svg">
</picture>
<!-- ![](./docs/zg_mnist_zg_torch_perf_0_speedupzigrad_pytorch_plotly.svg) -->

<sub>*Tensorflow excluded for scaling purposes (too slow).</sub>

The benchmark harness now lives in
[`benchmarks/`](./benchmarks/)
and emits machine-readable JSONL results with hardware and backend metadata,
plus a comparison tool for base-vs-candidate regression checks. The emitted
records now carry the checked-in spec path, declared benchmark provenance
(`data_source` plus preprocessing steps), CPU frequency policy when
discoverable, and captured host thread-environment hints alongside backend
telemetry. The current smoke suite covers deterministic primitive and BLAS
workloads, including conv-lowering coverage and a nested-broadcast matmul
fallback case, plus MNIST MLP, a char-level causal language model, pendulum
dynamics regression, CartPole-style DQN, and two-layer GCN workloads. The
harness now also includes a dedicated `compiler` suite for repeated eager
graph-session capture on the same maintained families, measuring
forward-plus-loss graph construction and teardown separately from model setup.
Host benchmark/build metadata now records the explicit BLAS provider as
`accelerate`, `openblas`, or `mkl`, and Zig runs also report host BLAS
dispatch telemetry so fallback usage is visible in the JSONL output:

```shell
zig build benchmark
zig build test-provider-parity
zig build test-example-smoke
zig build benchmark-primitive
zig build benchmark-blas
zig build benchmark-autograd
zig build benchmark-memory
zig build benchmark-compiler
zig build benchmark-models
zig build benchmark-validate
zig build test-benchmark-smoke
zig build test-benchmark-cuda-request-smoke
zig build test-benchmark-baseline-smoke
zig build test-benchmark-publication-smoke
zig build benchmark-publication-bundle -- --candidate-jsonl benchmarks/results/latest.jsonl --summary-output benchmarks/results/publication-summary.md --manifest-output benchmarks/results/publication-manifest.json
zig build benchmark-compare -- --baseline benchmarks/results/baseline.jsonl --candidate benchmarks/results/latest.jsonl
zig build benchmark-provider-report -- --input benchmarks/results/accelerate.jsonl --input benchmarks/results/openblas.jsonl --baseline-provider accelerate
zig build benchmark-thread-report -- --input benchmarks/results/thread-sweep.jsonl --baseline-thread-count 1
```

Optional PyTorch baseline execution is available per spec:

```shell
zig build benchmark -- --baseline pytorch
```

When a spec declares a baseline runner, the harness now treats that runner as
part of the JSONL contract: skipped baselines stay explicit, and launcher,
exit-code, or malformed-output failures produce structured `failed` records
instead of silently dropping the baseline row.

Compiler capture specs such as
[`benchmarks/specs/compiler/mnist-mlp-capture-synthetic.json`](./benchmarks/specs/compiler/mnist-mlp-capture-synthetic.json)
and
[`benchmarks/specs/compiler/gcn-capture-synthetic.json`](./benchmarks/specs/compiler/gcn-capture-synthetic.json)
reuse persistent model parameters while timing only forward-plus-loss graph
construction plus explicit teardown, which gives RFC-0001 a first executable
compiler-facing benchmark surface before RFC-0006 lazy tensors land:

```shell
zig build benchmark-compiler
zig build benchmark -- --spec benchmarks/specs/compiler/mnist-mlp-capture-synthetic.json --baseline pytorch --output .zig-cache/zigrad-compiler-capture.jsonl
zig build benchmark-validate -- --input .zig-cache/zigrad-compiler-capture.jsonl
```

Specs can now target a backend directly through a checked-in `device` field.
Host remains the default; CUDA-targeted specs such as
[`benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json`](./benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json)
and
[`benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json`](./benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json)
emit explicit `skipped` Zig records on non-CUDA builds or hosts instead of
aborting the harness, and successful CUDA runs include structured device
metadata under `backend.cuda`:

```shell
zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json --output .zig-cache/zigrad-benchmark-cuda.jsonl
zig build benchmark-validate -- --input .zig-cache/zigrad-benchmark-cuda.jsonl
```

`zig build test-benchmark-cuda-request-smoke` keeps that backend-aware contract
validated end to end. The `memory` suite remains host-only for now, and
PyTorch baseline rows for CUDA-targeted specs degrade into explicit `skipped`
records until a real CUDA PyTorch baseline path is wired in.

The harness now includes a dedicated contract validator. With no extra flags it
validates the committed spec tree; with `--input` it validates emitted JSONL
records against the referenced checked-in specs:

```shell
zig build benchmark-validate
zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output .zig-cache/zigrad-benchmark-validate.jsonl
zig build benchmark-validate -- --input .zig-cache/zigrad-benchmark-validate.jsonl
```

Thread sweeps do not require cloned spec files. The harness accepts repeated
`--thread-count` overrides and the comparison tool treats thread count as part
of the benchmark identity:

```shell
zig build benchmark -- \
  --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json \
  --thread-count 1 \
  --thread-count 2 \
  --thread-count 4 \
  --output benchmarks/results/thread-sweep.jsonl
zig build benchmark-thread-report -- \
  --input benchmarks/results/thread-sweep.jsonl \
  --baseline-thread-count 1 \
  --markdown-output benchmarks/results/thread-scaling.md \
  --json-output benchmarks/results/thread-scaling.json
```

The smoke suite also reports allocator and graph high-water marks for dedicated
memory benchmarks, including a tensor cache cycle and a synthetic MNIST
training step.

`zig build test-benchmark-smoke` exercises one checked-in spec per suite
through the real benchmark harness and fails if the validator detects contract
drift in the emitted JSONL artifact. `zig build
test-benchmark-baseline-smoke` extends that to the external baseline interface
by smoke-testing successful, malformed, and missing-runner cases. `zig build
test-benchmark-cuda-request-smoke` validates backend-aware CUDA request specs,
including explicit skip semantics on non-CUDA builds. `zig build
test-benchmark-publication-smoke` extends that coverage to the publication
surface by generating compare, provider-report, thread-report, and publication
bundle artifacts from smoke-scale inputs and rejecting empty or structurally
invalid outputs.

Provider-sensitive host BLAS correctness can be exercised independently with:

```shell
zig build test-provider-parity
zig build test-example-smoke
zig build test-provider-parity -Dhost_blas=openblas
zig build test-provider-parity -Dhost_blas=mkl -Dmkl_include_dir=/opt/intel/oneapi/mkl/latest/include -Dmkl_library_dir=/opt/intel/oneapi/mkl/latest/lib
zig build test-example-smoke -Dhost_blas=openblas
```

To turn multiple provider runs into publishable tables, generate provider-tagged
JSONL files and feed them into the host provider report step:

```shell
zig build benchmark -Dhost_blas=accelerate -- --output benchmarks/results/accelerate.jsonl
zig build benchmark -Dhost_blas=openblas -- --output benchmarks/results/openblas.jsonl
zig build benchmark -Dhost_blas=mkl -Dmkl_include_dir=/opt/intel/oneapi/mkl/latest/include -Dmkl_library_dir=/opt/intel/oneapi/mkl/latest/lib -- --output benchmarks/results/mkl.jsonl
zig build benchmark-provider-report -- \
  --input benchmarks/results/accelerate.jsonl \
  --input benchmarks/results/openblas.jsonl \
  --input benchmarks/results/mkl.jsonl \
  --baseline-provider accelerate \
  --markdown-output benchmarks/results/host-provider-report.md \
  --json-output benchmarks/results/host-provider-report.json
```

The report groups benchmarks by id and thread count, then emits Markdown and
JSON summaries with provider-vs-baseline latency deltas, speedups, memory
high-water marks, and host BLAS dispatch telemetry.

The thread-scaling report groups host records by benchmark id and provider,
then emits per-thread rows with baseline-relative deltas, speedups, and
scaling efficiency so RFC-0002 thread-behavior work can be validated from a
single provider run before cross-provider publication.

To package those outputs into a single manifest plus a human-readable summary,
point `benchmark-publication-bundle` at the emitted JSONL files and any derived
reports:

```shell
zig build benchmark-publication-bundle -- \
  --candidate-jsonl benchmarks/results/latest.jsonl \
  --baseline-jsonl benchmarks/results/baseline.jsonl \
  --extra-results-jsonl benchmarks/results/thread-sweep.jsonl \
  --comparison-json benchmarks/results/comparison.json \
  --comparison-text benchmarks/results/comparison.txt \
  --thread-report-json benchmarks/results/thread-scaling.json \
  --thread-report-markdown benchmarks/results/thread-scaling.md \
  --manifest-output benchmarks/results/publication-manifest.json \
  --summary-output benchmarks/results/publication-summary.md
```

The bundle validates that comparison and report artifacts still reference the
supplied JSONL inputs, records artifact sizes and runtime fingerprints, and
emits Markdown suitable for CI summaries or docs publication notes.

For runtime diagnostics outside the benchmark harness, set
`ZG_HOST_DIAGNOSTICS=1` when running an example or call
`cpu.runtimeDiagnostics()` / `cpu.writeRuntimeDiagnostics(...)` from your own
program. The snapshot reports the configured host provider, provider-specific
threading environment hints, and any observed batched-matmul fallback usage.

CUDA-enabled builds also expose runtime device metadata. Set
`ZG_CUDA_DIAGNOSTICS=1` when running a CUDA-capable example to log the selected
device id, device name, compute capability, multiprocessor count, total global
memory, and CUDA driver/runtime versions.

Benchmark authoring and smoke-policy guidance live in
[`benchmarks/AUTHORING.md`](./benchmarks/AUTHORING.md).

**Flexible**
Zigrad supports research workflows with high level abstractions for rapid prototyping, and integrations like Tensorboard and Mujoco. Zigrad supports the transition of research code to training infrastructure. 

Zigrad supports research through,

- Easy to use torch-like ergonomics
- A general purpose automatic differentiation system for n-dimensional data
- Eager execution and dynamic computation graph by default
- Computation graph tracing and visualization
- A design that naturally allows for custom differentiable operations

Zigrad supports engineering through,

- An architecture that enables deep control and customization through opt-in complexity,
- Offering flexible tradeoffs between performance characteristics like latency vs throughput
- Hardware-aware optimizations tailored to specific use cases and system requirements
- Fine-grained memory management and allocation control
- Cross-platform compatibility without compromising performance
- A streamlined design that avoids abstraction layers or build systems that hinder aggressive optimizations
<!-- Scalar API -->

## Features

### Trace the Computation Graph

![](./docs/comp_graph_mnist_simple_noag.svg)

An example of tracing the computation graph generated by a fully connected neural network for MNIST.

- *Input:* Batch of images 28x28 pixel samples.
- **Flatten:** `28x28 -> 784`
- **FC1**: Linear layer `784 -> 128`
- **ReLU**
- **FC2:** Linear layer `128 -> 64`
- **ReLU**
- **FC3:** Linear layer `64 -> 10`
- *Output:* Value for each of the 10 classes


We did not have to use Zigrad's modules to write this network at all, as Zigrad is backed by a capable autograd engine. Even when using the autograd backend to dynamically construct the same neural network Zigrad can still trace the graph and render it.

  > Note: Since the graph is generated from the autograd information, we set the labels for the nodes by naming the tensors for the sake of the diagram.

![](./docs/comp_graph_mnist_simple_ag.svg)

## Getting Started

Only dependency is a BLAS library.

CUDA remains optional and experimental. When an example supports it, build with
`-Denable_cuda=true` and select the runtime backend with
`ZG_DEVICE=host|cpu|cuda[:index]`.

### Linux

On linux (or intel mac) you have some options,

- MKL (recommended for best performance)
  - See https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html
  - Reccommend a system installation for simplicity although this can work with `conda` for example, just make sure you adjust the library paths as necessary.
  - Build with `zig build -Dhost_blas=mkl`
  - If headers or libraries are not on your default search path, add `-Dmkl_include_dir=/path/to/include -Dmkl_library_dir=/path/to/lib`
- OpenBLAS
  - See https://github.com/OpenMathLib/OpenBLAS/wiki/Precompiled-installation-packages
  - Likely available through your package manager as `libopenblas-dev` or `openblas-devel`
  - Build with `zig build -Dhost_blas=openblas`

### Apple Silicon

- Uses Accelerate by default.
- You can be explicit with `zig build -Dhost_blas=accelerate`.

### Examples

The `examples/` directory has some standalone templates you can take and modify, the zon files are pinned to commit hashes.

Hello world example shows how to run a backward pass using the `GraphManager.` Note that in this very simple example, we do not need the `GraphManager` and the script could be simplified but this is designed to get you familiar with the workflow.

```shell
git clone https://github.com/Marco-Christiani/zigrad/
cd zigrad/examples/hello-world
zig build run

# Optional CUDA runtime selection when built with CUDA enabled
ZG_DEVICE=cuda zig build run -Denable_cuda=true
```

Run the mnist demo

```shell
cd zigrad/examples/mnist
make help
make

# Optional CUDA smoke run when built with CUDA enabled
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```

Run the char-level language model demo

```shell
cd zigrad/examples/char-lm
zig build run

# Optional custom prompt for greedy generation
ZG_CHAR_LM_PROMPT="graph " zig build run

# Optional CUDA smoke run when built with CUDA enabled
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```

Runtime backend expectations are now explicit:

- `examples/hello-world`, `examples/mnist`, `examples/char-lm`,
  `examples/dqn`, and `examples/gcn` support `ZG_DEVICE=host|cpu|cuda[:index]`
  when built with `-Denable_cuda=true`.
- DQN and GCN now avoid host-only tensor reads in their runtime paths, but
  dedicated CUDA hardware validation is still pending on a GPU-capable runner.
- The char-level language model uses an embedded corpus and deterministic
  one-hot causal windows, so it runs from a clean checkout without downloads.

The maintained loss surface also avoids direct off-host tensor dereferences in
Zig now: `softmax_cross_entropy_loss`, `softmax`, `smooth_l1_loss`, and
`mse_loss` keep their fast host implementation, while non-host backends stage
through explicit host copies until dedicated device-native kernels land.

Fast smoke-mode entrypoints are also available for runtime validation without
full datasets or long training loops:

```shell
cd zigrad/examples/mnist
ZG_EXAMPLE_SMOKE=1 zig build run

cd zigrad/examples/char-lm
ZG_EXAMPLE_SMOKE=1 zig build run

cd zigrad/examples/dqn
ZG_EXAMPLE_SMOKE=1 zig build run

# Optional CUDA smoke run when built with CUDA enabled
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true

cd zigrad/examples/gcn
ZG_EXAMPLE_SMOKE=1 zig build run

# Optional CUDA smoke run when built with CUDA enabled
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```

## Vendored Dependencies

`safetensors-zg` is now vendored directly in this repository under `src/third_party/safetensors_zg/` with attribution and upstream licensing material in `third_party/safetensors_zg/`. The vendored source was imported from <https://github.com/Marco-Christiani/safetensors-zg> at commit `15787b35b541a4493630ec383750242eef422b64`.

`tensorboard/` also vendors the `zig-protobuf` runtime it builds against under `tensorboard/src/third_party/protobuf/`, with attribution and licensing material in `tensorboard/third_party/protobuf/`.

Run the vendored benchmark with:

```shell
zig build safetensors-benchmark
```

## Roadmap

A lot is planned and hoping for support from the Zig community so we can accomplish some of the more ambitious goals.

The detailed implementation plan now lives in
[`docs/roadmap.md`](./docs/roadmap.md),
with per-initiative RFCs in
[`docs/rfcs/`](./docs/rfcs/).

- More comprehensive MKL and CUDA support (in progress)
- Support for popular formats like ONNX and ggml.
- Standardized benchmarking procedures (always an ongoing effort)
- Lazy tensors
- Static graph optimization
- Dynamic graph compiler
- MLIR
- ZML translation for inference
- Apache TVM integration. [Github](https://github.com/apache/tvm/) [Homepage](https://tvm.apache.org)
- More examples like LLMs, physics and robotic control, etc.

## Known Issues and Limitations

- Documentation. As the API stabilizes more documentation will be added. For now, the examples are designed to be quickstart guides.
- Effort has been directed towards performant primitives, not many layer types have been implemented
  - e.g. conv, pooling, etc are test implementations for verification, they are slow and unoptimized, I would not use them

## Contributing

- [Join the discord](https://discord.gg/JWSSfWj3Uf) and into the dev channels
- Any open issue is available for development, just leave a comment mentioning your interest and I can provide support to help get you started if necessary
- Otherwise, **please open an issue first, before working on a PR**
- If you are interested in contributing but do not know where to start then open an issue or leave a comment
