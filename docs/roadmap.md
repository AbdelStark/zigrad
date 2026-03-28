# Zigrad Roadmap and RFC Index

This document decomposes the public roadmap into implementation-ready RFCs and
specifications. It should be treated as the canonical planning index for major
platform, compiler, interoperability, and example work.

The older [roadmap.norg](./roadmap.norg)
remains useful historical context. The files listed here are the new execution
documents we will implement against.

## Status Legend

- `Draft`: scoped, but further validation is required before coding.
- `Ready`: sufficiently specified to begin implementation.
- `Planned`: accepted into the roadmap, but intentionally sequenced behind
  dependencies.
- `Exploratory`: worth pursuing, but requires spikes or external validation.
- `Blocked`: accepted, but cannot start until predecessor RFCs land.

## Program Principles

- Preserve Zigrad's research-first, eager-by-default workflow.
- Add complexity only where it unlocks measurable capability or performance.
- Prefer staged delivery over large rewrites hidden behind a single branch.
- Make every performance claim reproducible through the benchmark program.
- Keep optional integrations optional at build time and explicit at runtime.
- Treat interoperability as product surface, not just import/export plumbing.

## Workstreams

### Phase 0: Measurement and Backend Foundations

- [RFC-0001](./rfcs/0001-benchmarking-program.md)
  defines the benchmark harness, regression policy, and result schema.
- [RFC-0002](./rfcs/0002-onemkl-host-backend.md)
  expands the CPU backend story around oneMKL and provider selection.
- [RFC-0003](./rfcs/0003-cuda-backend.md)
  brings CUDA from experimental to supported.

### Phase 1: Interop and Execution Model

- [RFC-0004](./rfcs/0004-onnx-interop.md)
  covers ONNX import/export.
- [RFC-0005](./rfcs/0005-ggml-gguf-interop.md)
  covers ggml/GGUF model and weight interoperability.
- [RFC-0006](./rfcs/0006-lazy-tensors.md)
  introduces deferred execution as an opt-in execution mode.
- [RFC-0012](./rfcs/0012-examples-and-reference-models.md)
  defines the reference example portfolio that will validate the stack.

### Phase 2: Optimization and Compilation

- [RFC-0007](./rfcs/0007-static-graph-optimization.md)
  defines a verifiable optimization pipeline for captured graphs.
- [RFC-0008](./rfcs/0008-dynamic-graph-compiler.md)
  adds specialization and compilation for dynamic graphs.
- [RFC-0009](./rfcs/0009-mlir-lowering-pipeline.md)
  introduces MLIR as an optional lowering and interchange layer.

### Phase 3: Inference Translation and External Compilers

- [RFC-0010](./rfcs/0010-zml-inference-bridge.md)
  defines inference-only translation into ZML.
- [RFC-0011](./rfcs/0011-apache-tvm-integration.md)
  defines optional TVM integration for ahead-of-time and autotuned execution.

## RFC Matrix

| ID | Title | Status | Priority | Depends on | Notes |
| --- | --- | --- | --- | --- | --- |
| RFC-0001 | Standardized Benchmarking Program | `Ready` | P0 | None | Harness, JSONL output, comparison/regression tooling, authoring guide, smoke CI, synthetic BLAS/autograd/memory/MNIST/DQN/GCN plus conv-lowering and broadcast-fallback matmul coverage, and host thread-sweep/scaling-report workflows are landed; Zig JSONL output now carries host BLAS dispatch telemetry, while future CUDA/compiler/interop suites remain. |
| RFC-0002 | oneMKL Host Backend | `Ready` | P0 | RFC-0001 | Explicit host BLAS provider selection, nested batched-matmul broadcast correctness, host dense-dispatch telemetry, benchmark-visible fallback telemetry, example-model audit coverage, legacy Conv2D lowering audit, a provider-sensitive numerical parity suite, opt-in runtime diagnostics hooks, example runtime smoke coverage for hello-world/MNIST/DQN/GCN, and Markdown/JSON provider plus thread-scaling report generators are landed; oneMKL execution and published provider comparison runs remain. |
| RFC-0003 | CUDA Backend | `Ready` | P0 | RFC-0001 | Turns experimental CUDA into a supported execution backend. |
| RFC-0004 | ONNX Interop | `Planned` | P1 | RFC-0001, RFC-0007 | Best treated as import/export on top of a stable graph IR. |
| RFC-0005 | ggml/GGUF Interop | `Planned` | P1 | RFC-0001, RFC-0012 | Critical for LLM examples and inference compatibility. |
| RFC-0006 | Lazy Tensors | `Planned` | P1 | RFC-0001, RFC-0002, RFC-0003 | Introduces capture without breaking eager execution. |
| RFC-0007 | Static Graph Optimization | `Planned` | P1 | RFC-0006 | First optimization layer and foundation for compiler work. |
| RFC-0008 | Dynamic Graph Compiler | `Draft` | P2 | RFC-0006, RFC-0007 | Specialization and caching for dynamic workloads. |
| RFC-0009 | MLIR Lowering Pipeline | `Exploratory` | P2 | RFC-0007, RFC-0008 | Optional compiler interoperability layer. |
| RFC-0010 | ZML Inference Bridge | `Draft` | P2 | RFC-0007 | Enables inference handoff to ZML for pure serving flows. |
| RFC-0011 | Apache TVM Integration | `Exploratory` | P3 | RFC-0001, RFC-0007, RFC-0009 | External compiler/autotuning path. |
| RFC-0012 | Examples and Reference Models | `Planned` | P1 | RFC-0001, RFC-0002, RFC-0003 | Program-level validation for all major capabilities. |

## Recommended Implementation Order

1. RFC-0001 Standardized Benchmarking Program
2. RFC-0002 oneMKL Host Backend
3. RFC-0003 CUDA Backend
4. RFC-0012 Examples and Reference Models, initial CV/RL refresh
5. RFC-0006 Lazy Tensors
6. RFC-0007 Static Graph Optimization
7. RFC-0004 ONNX Interop
8. RFC-0005 ggml/GGUF Interop
9. RFC-0010 ZML Inference Bridge
10. RFC-0008 Dynamic Graph Compiler
11. RFC-0009 MLIR Lowering Pipeline
12. RFC-0011 Apache TVM Integration

## Definition of Done for the Roadmap Program

The roadmap is considered complete only when:

- All RFCs have been implemented or explicitly closed with replacement plans.
- Every accepted feature has correctness tests and benchmark coverage.
- CPU and CUDA execution paths are both validated on representative models.
- Import/export and translation paths have round-trip or conformance coverage.
- At least one LLM, one physics/control example, and one RL example are kept
  green in CI smoke form.
- The benchmark program can reproduce published performance claims from a clean
  checkout.

## RFC Authoring Conventions

Every RFC in this folder set must maintain:

- an explicit status,
- concrete dependency and blocking relationships,
- a phased delivery plan,
- measurable acceptance criteria,
- test and benchmark requirements,
- a section describing what will not be done in the RFC.

## Agentic Context

### RFC-0001 2026-03-28 Benchmark Provenance Contract

- Completed:
  - Required every committed benchmark spec to declare machine-readable
    provenance (`data_source` plus preprocessing steps) and threaded that
    provenance into emitted JSONL records alongside the originating `spec_path`.
  - Extended benchmark runtime metadata to capture CPU frequency policy when
    discoverable plus host thread-environment hints, and mirrored the same
    fields into optional PyTorch baseline records.
  - Updated benchmark authoring and README documentation so the reproducibility
    contract now matches the shipped harness behavior.
- Remains:
  - Apply the same provenance discipline to future dataset-backed and
    accelerator-backed suites once those RFCs land executable workloads.
- Blockers:
  - This macOS run could not populate Linux-only CPU governor metadata, so the
    field validated as optional rather than populated.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output /tmp/zigrad-benchmark-provenance.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/blas/dot-f32-262144.json --baseline pytorch --thread-count 2 --output /tmp/zigrad-benchmark-baseline-provenance.jsonl`

### RFC-0002 2026-03-28 Thread Scaling Workflow

- Completed:
  - Added repeatable `--thread-count <n>` benchmark overrides in
    [`benchmarks/src/cli.zig`](../benchmarks/src/cli.zig) so host-provider
    performance work can execute deterministic thread sweeps without cloning
    spec files.
  - Added
    [`benchmarks/src/thread_report.zig`](../benchmarks/src/thread_report.zig)
    plus
    [`benchmarks/src/thread_report_main.zig`](../benchmarks/src/thread_report_main.zig)
    and the `zig build benchmark-thread-report` entrypoint in
    [`build.zig`](../build.zig), producing Markdown/JSON scaling summaries with
    baseline-relative speedup and efficiency columns.
  - Updated
    [`benchmarks/src/compare.zig`](../benchmarks/src/compare.zig)
    and
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../benchmarks/runners/pytorch/mnist_mlp.py)
    so thread-swept JSONL files compare cleanly and baseline records preserve
    the same thread metadata.
  - Documented the workflow in
    [`README.md`](../README.md),
    [`benchmarks/README.md`](../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../benchmarks/AUTHORING.md).
- Remains:
  - Run the same thread-sweep benchmark groups on OpenBLAS and oneMKL hosts and
    publish the first provider-specific scaling tables.
  - Decide whether CI should archive scaling Markdown/JSON alongside provider
    comparison artifacts once cross-provider runners are available.
- Blockers:
  - This machine still only exposes the Accelerate backend, so the new workflow
    validated functionally but did not yet produce OpenBLAS/oneMKL scaling
    data.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark-thread-report -- --help`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json --thread-count 1 --thread-count 2 --output /tmp/zigrad-thread-sweep.jsonl`
  - `zig build benchmark-thread-report -- --input /tmp/zigrad-thread-sweep.jsonl --baseline-thread-count 1 --markdown-output /tmp/zigrad-thread-scaling.md --json-output /tmp/zigrad-thread-scaling.json`

### RFC-0001 2026-03-28 Host Thread Scaling Workflow

- Completed:
  - Extended the benchmark harness so repeated `--thread-count` overrides
    duplicate selected specs at runtime and keep the checked-in JSON manifest
    surface stable.
  - Added the dedicated host thread-scaling report workflow and updated the
    comparison tool so thread count is part of the benchmark record identity.
- Remains:
  - Extend the same sweep/report workflow to future CUDA and compiler suites as
    those RFCs become executable.
  - Capture non-skipped PyTorch baseline data on a machine with `torch`
    installed.
- Blockers:
  - Local validation still only exercised the macOS Accelerate backend, so
    cross-provider scaling output remains pending.
- Validation:
  - `zig build test`
  - `zig build benchmark-compare -- --baseline /tmp/zigrad-thread-sweep.jsonl --candidate /tmp/zigrad-thread-sweep.jsonl --runner zig`

### RFC-0001 2026-03-28 Conv Lowering Coverage

- Completed:
  - Added a reusable legacy conv lowering helper in
    [`src/nn/conv_utils.zig`](../src/nn/conv_utils.zig) that benchmarks can
    call directly via `im2col` plus batched matmul.
  - Extended the `blas` suite manifest and workload coverage to support
    deterministic `blas_conv2d_im2col` specs with stride/padding/dilation
    fields and added two checked-in conv benchmark specs under
    [`benchmarks/specs/blas/`](../benchmarks/specs/blas/).
  - Expanded the optional PyTorch baseline runner and benchmark execution tests
    to cover the new conv workload kind.
- Remains:
  - Add CUDA/compiler/interop benchmark suites as those RFCs become executable.
  - Capture non-skipped PyTorch baseline data on a machine with `torch`
    installed.
- Blockers:
  - No local `torch` installation was available, so the baseline path validated
    only the explicit skip behavior.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --output /tmp/zigrad-conv-benchmark.jsonl`
  - `zig build benchmark-blas -- --output /tmp/zigrad-blas.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --baseline pytorch --output /tmp/zigrad-conv-benchmark-baseline.jsonl`

### RFC-0002 2026-03-28 Provider Report Generator

- Completed:
  - Added a dedicated `benchmark-provider-report` build step backed by
    [`benchmarks/src/provider_report.zig`](../benchmarks/src/provider_report.zig)
    and [`benchmarks/src/provider_report_main.zig`](../benchmarks/src/provider_report_main.zig)
    so multiple host-provider benchmark JSONL files can be consolidated into
    Markdown and JSON reports.
  - The report groups host-device records by benchmark id and thread count,
    computes provider-vs-baseline latency deltas and speedups, and carries
    forward memory high-water marks plus host BLAS dispatch telemetry.
  - Updated benchmark and top-level docs to make provider benchmarking an
    explicit workflow instead of an ad hoc manual comparison step.
- Remains:
  - Run the same benchmark groups on OpenBLAS and oneMKL hosts and publish the
    first real provider comparison tables generated by the new report step.
  - Decide whether scheduled CI should archive the generated Markdown/JSON
    provider reports once Linux/x86 providers are routinely available.
- Blockers:
  - This machine still only exercised the Accelerate backend, so the new report
    generator validated with unit fixtures and single-provider real output
    rather than actual multi-provider x86 runs.
- Validation:
  - `zig build test`
  - `zig build benchmark-provider-report -- --help`

### RFC-0001 2026-03-28 Host Dispatch Telemetry Promotion

- Completed:
  - Extended the benchmark result schema so Zig benchmark records can carry
    per-workload host BLAS telemetry, including direct batched-dispatch counts
    and nested-broadcast fallback counts.
  - Un-ignored `benchmarks/specs/**/*.json` in the repository so benchmark
    specs become committed harness inputs instead of hidden local-only state.
  - Added a deterministic primitive benchmark spec for nested-broadcast matmul
    so the smoke suite now covers the manual fallback path introduced for
    non-modulo-safe batch broadcasts.
  - Threaded the telemetry through the benchmark harness and documented the new
    backend metadata in the benchmark README/authoring guide and top-level
    README.
- Remains:
  - Add equivalent benchmark-visible telemetry for CUDA/compiler/interop
    backends when those RFCs become executable.
  - Collect non-skipped PyTorch baseline data on a machine with `torch`
    installed.
- Blockers:
  - Local validation still exercised only the macOS Accelerate backend, so
    OpenBLAS/oneMKL benchmark records with the new telemetry remain pending.
- Validation:
  - `zig build test`
  - `zig build benchmark-primitive -- --output /tmp/zigrad-primitive.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/matmul-broadcast-fallback-f32-2x2x2x3-2x1x3x2.json --output /tmp/zigrad-broadcast-fallback.jsonl`
  - `zig build benchmark-models -- --output /tmp/zigrad-models.jsonl`

### RFC-0001 Snapshot

- Completed:
  - Added the benchmark harness under [`benchmarks/`](../benchmarks/) with JSON
    spec loading, JSON-lines result emission, runtime/system/backend metadata
    capture, and build entrypoints.
  - Landed initial `primitive`, `model-train`, and `model-infer` specs covering
    deterministic add, deterministic matmul, synthetic MNIST-style MLP train,
    synthetic MNIST-style MLP infer, synthetic CartPole-style DQN train/infer,
    and synthetic two-layer GCN train/infer.
  - Added an optional PyTorch baseline runner that emits `skipped` records when
    `torch` is unavailable and now understands the expanded model benchmark
    kinds.
  - Added benchmark smoke CI in
    [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml).
  - Added benchmark result comparison tooling with threshold-based regression
    classification, `zig build benchmark-compare`, and JSON/text comparison
    outputs for local runs and CI.
  - Added the benchmark authoring guide in
    [`benchmarks/AUTHORING.md`](../benchmarks/AUTHORING.md).
  - Updated smoke CI to benchmark the base revision and current checkout on the
    same runner, compare results, and upload comparison artifacts.
- Remains:
  - Cross-platform baseline data with `torch` installed and broader
    CUDA/compiler/interop workload coverage.
  - Published benchmark reporting pages or artifacts beyond uploaded CI JSONL.
- Blockers:
  - No local PyTorch installation was present in this run, so baseline execution
    validated only the explicit `skipped` path.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark-models`
  - `zig build benchmark-compare -- --help`
  - `zig build benchmark`
  - `zig build benchmark-models -- --baseline pytorch`
  - `zig build benchmark-compare -- --baseline benchmarks/results/latest.jsonl --candidate benchmarks/results/latest.jsonl --runner zig --json-output benchmarks/results/comparison.json --report-output benchmarks/results/comparison.txt`
  - `zig build benchmark-primitive`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`
  - `python3 -c "import json, pathlib; print(pathlib.Path('benchmarks/results/comparison.json').exists())"`

### RFC-0001 2026-03-27 BLAS + Autograd Coverage

- Completed:
  - Added dedicated `blas` and `autograd` benchmark suites with deterministic
    dot and matvec workloads under [`benchmarks/specs/`](../benchmarks/specs/).
  - Extended the harness manifest, CLI grouping, and `zig build` entrypoints
    so `benchmark`, `benchmark-blas`, and `benchmark-autograd` execute the new
    coverage.
  - Added Zig workload implementations for BLAS forward coverage and autograd
    backward coverage in
    [`benchmarks/src/workload.zig`](../benchmarks/src/workload.zig).
  - Expanded the optional PyTorch baseline runner to emit comparable records
    for the new BLAS and autograd benchmark kinds.
- Remains:
  - Add CUDA-targeted suites and backend-specific parity checks once RFC-0003
    work begins.
  - Add compiler, interop, and memory suites to cover the remaining RFC-0001
    benchmark taxonomy.
- Blockers:
  - No local `torch` install was available during this run, so PyTorch parity
    was validated through runner compilation and skip-path behavior rather than
    executed framework comparisons.
- Validation:
  - `zig build test`
  - `zig build benchmark-blas`
  - `zig build benchmark-autograd`
  - `zig build benchmark`
  - `zig build benchmark -- --baseline pytorch --group blas`
  - `zig build benchmark -- --baseline pytorch --group autograd`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`

### RFC-0001 2026-03-27 Memory Coverage

- Completed:
  - Added host caching allocator telemetry and graph arena capacity reporting
    so benchmark records can include memory high-water marks.
  - Added a dedicated `memory` suite with a tensor cache cycle workload and a
    synthetic MNIST training-step workload.
  - Extended benchmark comparison output to include memory deltas and fail on
    threshold-exceeding memory regressions.
- Remains:
  - Extend the same benchmark taxonomy to CUDA memory accounting and future
    compiler/interop workloads.
- Blockers:
  - PyTorch baseline coverage still does not apply to Zigrad-native memory
    telemetry workloads.
- Validation:
  - `zig build test`
  - `zig build benchmark-memory`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/memory.jsonl --candidate benchmarks/results/memory.jsonl --runner zig --json-output benchmarks/results/memory-comparison.json --report-output benchmarks/results/memory-comparison.txt`

### RFC-0002 2026-03-27 Provider Selection + Metadata

- Completed:
  - Added `HostBlasProvider` and `HostBackendInfo` under
    [`src/device/host_blas_provider.zig`](../src/device/host_blas_provider.zig)
    and exposed the configured provider through the host backend/public device
    API.
  - Replaced the build-graph `-Denable_mkl` toggle with explicit
    `-Dhost_blas=auto|accelerate|openblas|mkl` selection, preserving
    `-Denable_mkl=true` as a compatibility alias for `mkl`.
  - Added oneMKL include/library override flags and updated docs/CI commands to
    use explicit provider selection.
  - Updated benchmark metadata to emit `accelerate`, `openblas`, or `mkl`
    instead of the ambiguous Linux `blas` label.
- Remains:
  - Validate Linux OpenBLAS and oneMKL builds, then add cross-provider
    numerical parity tests and benchmark tables.
  - Audit provider-backed conv, linear, and batched-GEMM execution paths.
- Blockers:
  - Local validation ran only on macOS/Accelerate, so the Linux OpenBLAS and
    oneMKL paths remain unexecuted in this run.
- Validation:
  - `zig build test`
  - `zig build -Dhost_blas=accelerate benchmark`
  - `python3 - <<'PY'`
    `import json`
    `from pathlib import Path`
    `first = json.loads(Path("benchmarks/results/latest.jsonl").read_text().splitlines()[0])`
    `print(first["backend"]["host_provider"])`
    `PY`

### RFC-0002 2026-03-28 Host Provider Numerical Parity Suite

- Completed:
  - Added shared seeded-fixture and tolerance helpers in
    [`benchmarks/src/test_support.zig`](../benchmarks/src/test_support.zig)
    so provider-facing benchmark tests reuse deterministic data generation.
  - Added
    [`benchmarks/src/provider_parity.zig`](../benchmarks/src/provider_parity.zig)
    with numeric parity coverage for host GEMV alpha/beta semantics,
    direct batched `bmm_acc`, nested-broadcast batched matmul
    forward/backward, and legacy Conv2D im2col lowering against pure reference
    math.
  - Fixed [`src/nn/conv_utils.zig`](../src/nn/conv_utils.zig) so multi-channel
    Conv2D bias is applied across the full spatial slice for each output
    channel rather than cycling incorrectly through a flat broadcast.
  - Added `zig build test-provider-parity` in
    [`build.zig`](../build.zig) and wired the same suite into
    [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml)
    under `-Dhost_blas=openblas`, giving RFC-0002 a reusable Linux/OpenBLAS
    correctness gate instead of only macOS-local validation.
- Remains:
  - Execute the parity suite on oneMKL-configured Linux/x86 hardware.
  - Publish OpenBLAS vs oneMKL provider comparison tables for representative
    models once both environments are available.
  - Land runtime logging hooks for provider/fallback mode outside test and
    benchmark surfaces.
- Blockers:
  - Local execution still only had the macOS Accelerate backend, so the new
    suite was validated under one provider locally and prepared for OpenBLAS CI
    rather than exercised across all target providers in this run.
- Validation:
  - `zig build test-provider-parity`
  - `zig build test`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`

### RFC-0002 2026-03-28 Runtime Diagnostics Hooks

- Completed:
  - Added public host runtime diagnostics helpers in
    [`src/device/host_device.zig`](../src/device/host_device.zig) covering the
    configured provider, provider-specific thread environment hints, BLAS call
    counters, and batched-matmul fallback-mode summaries.
  - Re-exported the diagnostics API through [`src/device.zig`](../src/device.zig),
    [`src/device/root.zig`](../src/device/root.zig), and
    [`src/zigrad.zig`](../src/zigrad.zig) so examples and downstream code can
    call `zg.device.hostDiagnosticsEnabled()`,
    `zg.device.configuredRuntimeDiagnostics()`, and
    `cpu.writeRuntimeDiagnostics(...)`.
  - Wired opt-in diagnostics logging into the hello-world, MNIST, DQN, and GCN
    example entrypoints behind `ZG_HOST_DIAGNOSTICS=1`, giving RFC-0002 a
    user-facing runtime debug path outside the benchmark harness.
- Remains:
  - Execute the same diagnostics surface on Linux OpenBLAS and oneMKL hosts.
  - Publish OpenBLAS vs oneMKL benchmark tables once both providers have
    representative model results.
  - Decide whether to add a structured JSON runtime diagnostics surface later.
- Blockers:
  - Local validation still only exercised the macOS Accelerate backend, so the
    new diagnostics were not observed on Linux/x86 OpenBLAS or oneMKL in this
    run.
- Validation:
  - `zig build test`
  - `cd examples/hello-world && ZG_HOST_DIAGNOSTICS=1 zig build run`
  - `cd examples/mnist && zig build -Dhost_blas=accelerate`
  - `cd examples/dqn && zig build -Dhost_blas=accelerate`
  - `cd examples/gcn && zig build -Dhost_blas=accelerate`

### RFC-0002 2026-03-28 Example Runtime Smoke Suite

- Completed:
  - Added configurable smoke execution paths to
    [`examples/mnist/src/main.zig`](../examples/mnist/src/main.zig),
    [`examples/dqn/src/dqn_train.zig`](../examples/dqn/src/dqn_train.zig),
    [`examples/dqn/src/main.zig`](../examples/dqn/src/main.zig), and
    [`examples/gcn/src/main.zig`](../examples/gcn/src/main.zig) while
    preserving their default full-workload behavior; MNIST now uses bundled
    small CSVs without touching `mnist.stz`, DQN runs a bounded replay-buffer
    training loop, and GCN can execute against a new synthetic in-memory graph
    fixture from [`examples/gcn/src/dataset.zig`](../examples/gcn/src/dataset.zig).
  - Added a repo-level `zig build test-example-smoke` runner in
    [`build.zig`](../build.zig) backed by
    [`tests/src/example_smoke_main.zig`](../tests/src/example_smoke_main.zig),
    so hello-world, MNIST, DQN, and GCN all execute real runtime paths as part
    of the default `zig build test` surface.
  - Fixed
    [`examples/dqn/src/replay_buffer.zig`](../examples/dqn/src/replay_buffer.zig)
    so `sample2` no longer treats zero-initialized bookkeeping slots as
    already-consumed indices, which previously made full-buffer
    without-replacement sampling hang in small deterministic runs.
  - Wired the same smoke gate into
    [`.github/workflows/benchmark-smoke.yml`](../.github/workflows/benchmark-smoke.yml)
    under `-Dhost_blas=openblas`, and documented the new validation commands in
    [`README.md`](../README.md).
- Remains:
  - Execute the same example smoke suite on oneMKL-configured Linux/x86
    hardware.
  - Decide whether future RFC-0012 example artifacts should extend this smoke
    runner with checkpoint and downloadable-dataset validation.
- Blockers:
  - Local validation still only exercised the macOS Accelerate backend, so the
    new suite is wired for Linux/OpenBLAS CI but not yet observed on OpenBLAS
    or oneMKL locally.
- Validation:
  - `zig build test-example-smoke`
  - `zig build test`
  - `cd examples/hello-world && zig build run`
  - `cd examples/mnist && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `cd examples/dqn && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `cd examples/gcn && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`

### RFC-0002 2026-03-28 Batched GEMM Broadcast Correctness

- Completed:
  - Fixed nested batch-broadcast indexing in
    [`src/ndarray.zig`](../src/ndarray.zig) so broadcasted batched matmul no
    longer relies on incorrect flatten-and-modulo mapping.
  - Added a per-batch `matmul` fallback for non-modulo-safe layouts while
    keeping the direct batched dispatch fast path for safe host/CUDA cases.
  - Fixed accumulation into smaller broadcast-compatible outputs and forwarded
    `alpha`/`beta` through [`src/ndtensor.zig`](../src/ndtensor.zig), which
    makes broadcasted batched-matmul backward passes accumulate correctly.
  - Added forward/backward regression coverage for the `[2,2,2,3] x [2,1,3,2]`
    case and propagated `-Dhost_blas=...` through the example build entrypoints
    under [`examples/`](../examples/).
  - Updated [`examples/gcn/src/main.zig`](../examples/gcn/src/main.zig) to the
    current `std.json.Stringify` API so the GCN example builds again.
- Remains:
  - Add Linux OpenBLAS/oneMKL numerical parity coverage and benchmark tables.
  - Audit conv and linear example/runtime paths beyond the batched matmul core.
  - Execute the new runtime smoke coverage on Linux OpenBLAS and oneMKL
    builds.
- Blockers:
  - No Linux OpenBLAS or oneMKL runtime was available in this run.
- Validation:
  - `zig build test`
  - `cd examples/hello-world && zig build -Dhost_blas=accelerate`
  - `cd examples/mnist && zig build -Dhost_blas=accelerate`
  - `cd examples/dqn && zig build -Dhost_blas=accelerate`
  - `cd examples/gcn && zig build -Dhost_blas=accelerate`

### RFC-0002 2026-03-28 Dense Dispatch Audit + Example Graph Injection

- Completed:
  - Added host BLAS operation telemetry in
    [`src/device/host_device.zig`](../src/device/host_device.zig) for `dot`,
    `matvec`, `matmul`, and `bmm_acc`, plus public exports through
    [`src/device.zig`](../src/device.zig) and
    [`src/zigrad.zig`](../src/zigrad.zig).
  - Added benchmark-side regression coverage in
    [`benchmarks/src/provider_audit.zig`](../benchmarks/src/provider_audit.zig)
    that asserts exact host `matmul`/`bmm_acc` counts for the MNIST, DQN, and
    GCN example forward paths.
  - Wired the benchmark test module to import the example model sources through
    [`build.zig`](../build.zig) so the audit runs against the example
    implementations rather than synthetic copies.
  - Added explicit-graph construction hooks to the example model initializers in
    [`examples/mnist/src/model.zig`](../examples/mnist/src/model.zig),
    [`examples/dqn/src/dqn_model.zig`](../examples/dqn/src/dqn_model.zig), and
    [`examples/gcn/src/model.zig`](../examples/gcn/src/model.zig), which makes
    these paths reproducible in tests without depending on the global graph.
  - Fixed a latent no-grad `scatter_add` offset leak in
    [`src/ndtensor.zig`](../src/ndtensor.zig) and updated the GCN example to
    pass explicit graph handles for temporary tensors created during message
    propagation.
- Remains:
  - Validate OpenBLAS and oneMKL parity on Linux/x86 hardware.
  - Audit the legacy reference Conv2D path separately; it still is not routed
    through provider-backed GEMM lowering.
  - Decide whether host op telemetry should eventually flow into benchmark JSONL
    artifacts instead of staying as a test/debug surface.
- Blockers:
  - This run still had no Linux OpenBLAS/oneMKL environment, so provider parity
    remains unexecuted locally.
- Validation:
  - `zig build test`
  - `zig build benchmark-models`
  - `cd examples/mnist && zig build -Dhost_blas=accelerate`
  - `cd examples/dqn && zig build -Dhost_blas=accelerate`
  - `cd examples/gcn && zig build -Dhost_blas=accelerate`

### RFC-0002 2026-03-28 Legacy Conv2D BLAS Audit

- Completed:
  - Added `conv2dOutputShape` and `conv2dForwardIm2col` in
    [`src/nn/conv_utils.zig`](../src/nn/conv_utils.zig), which routes the
    legacy reference conv path through provider-backed batched matmul lowering.
  - Added exact telemetry regression coverage in
    [`benchmarks/src/provider_audit.zig`](../benchmarks/src/provider_audit.zig)
    proving the legacy conv path issues the expected host `bmm_acc` and
    per-batch `matmul` calls.
  - Added benchmark coverage for the same lowering path under
    [`benchmarks/specs/blas/`](../benchmarks/specs/blas/) and extended the
    PyTorch baseline runner to understand the new spec kind.
- Remains:
  - Validate the conv audit on Linux OpenBLAS and oneMKL builds.
  - Add cross-provider parity checks and published comparison tables.
- Blockers:
  - Only the macOS Accelerate path was available locally, and PyTorch was not
    installed, so parity and executed baseline comparisons remain pending.
- Validation:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --output /tmp/zigrad-conv-benchmark.jsonl`
  - `zig build benchmark-blas -- --output /tmp/zigrad-blas.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --baseline pytorch --output /tmp/zigrad-conv-benchmark-baseline.jsonl`
