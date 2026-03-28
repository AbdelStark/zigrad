# RFC-0001: Standardized Benchmarking Program

Status: `Ready`  
Priority: `P0`  
Depends on: None  
Blocks: RFC-0002, RFC-0003, RFC-0004, RFC-0005, RFC-0006, RFC-0012  
Last updated: `2026-03-28`

## Summary

Zigrad needs a first-class benchmarking program that measures kernels, model
training, model inference, compiler passes, memory behavior, and cross-framework
comparisons in a reproducible way. This RFC defines the benchmark harness,
result schema, comparison policy, CI strategy, and publication process that all
future optimization work will use.

## Motivation

The roadmap contains multiple performance-sensitive initiatives: oneMKL, CUDA,
lazy execution, graph optimization, dynamic compilation, MLIR lowering, ZML
translation, and TVM integration. Without a consistent benchmark program, it
will be impossible to determine whether these efforts actually improve latency,
throughput, memory use, or developer ergonomics.

The current project has ad hoc performance artifacts in `docs/`, but it does
not yet define:

- a stable suite layout,
- required hardware metadata,
- baseline frameworks,
- noise controls,
- regression thresholds, or
- a result format suitable for automation.

## Goals

- Provide a single benchmark entrypoint and directory layout.
- Cover microbenchmarks, model benchmarks, and end-to-end workflows.
- Store benchmark results in a machine-readable format.
- Make framework comparisons reproducible across machines and commits.
- Support both local exploration and CI-driven regression detection.
- Record hardware, backend, dtype, dataset, and workload metadata alongside
  every result.

## Non-Goals

- Building a benchmark web service in the first iteration.
- Replacing existing docs images immediately.
- Enforcing exact cross-machine performance comparability.
- Running the full benchmark suite on every PR.

## Benchmark Taxonomy

The program will define the following suites:

- `primitive`: scalar ops, elementwise ops, reductions, broadcasting, reshape,
  transpose, gather, indexing, and memory copies.
- `blas`: GEMM, batched GEMM, matvec, conv lowering where relevant.
- `autograd`: forward+backward for representative graphs and module stacks.
- `model-train`: MNIST MLP/CNN, DQN, GCN, and future reference models.
- `model-infer`: eager inference, lazy inference, translated inference.
- `compiler`: graph capture, optimization pass cost, compile latency, cache hit
  behavior.
- `interop`: ONNX import cost, GGUF load cost, ZML translation cost.
- `memory`: peak bytes, allocator churn, host-device transfer volume.

## Proposed Layout

```text
benchmarks/
  README.md
  specs/
    primitive/
    compiler/
    model-train/
    model-infer/
  runners/
    zig/
    pytorch/
    zml/
  datasets/
    manifests/
  results/
    .gitignore
  scripts/
```

`zig build benchmark` will dispatch into benchmark suites. Additional steps such
as `zig build benchmark-primitive`, `zig build benchmark-compiler`, and
`zig build benchmark-models` may be added for convenience.

## Result Schema

Every result record must contain:

- benchmark id,
- git commit SHA,
- dirty tree flag,
- Zig version,
- benchmark harness version,
- operating system and kernel version,
- CPU model, core count, frequency policy,
- GPU model, driver version, compute capability if applicable,
- backend configuration: Accelerate/OpenBLAS/oneMKL/CUDA/etc,
- dtype,
- tensor shapes,
- batch size,
- warmup count,
- measured iterations,
- min/median/mean/p95/max latency,
- throughput where relevant,
- peak memory where relevant,
- notes for skipped or unsupported cases.

The initial on-disk format will be JSON lines. CSV export can be layered on top.

## Baseline Framework Policy

The benchmarking program will compare Zigrad against:

- PyTorch for CPU and CUDA where feature parity exists,
- ZML for inference-oriented workflows once the bridge exists,
- optionally TensorFlow or JAX only when a benchmark specifically requires it.

Baseline runners must record:

- framework version,
- exact model definition,
- backend flags,
- dtype,
- optimizer settings,
- dataset preprocessing choices.

## Reproducibility Rules

- Random seeds must be fixed and recorded.
- Benchmark specs must declare data source and preprocessing.
- CPU benchmarks must support thread count pinning.
- CUDA benchmarks must separate compile/warmup from measurement.
- Benchmarks that allocate model weights must distinguish one-time setup cost
  from steady-state iteration cost.
- Results used in docs must be reproducible from the published spec.

## CI Strategy

PR CI will run a smoke subset:

- one primitive suite,
- one CPU model training suite,
- one inference suite,
- one correctness-checked performance sanity benchmark.

Scheduled CI will run the broader suite on dedicated runners where available.
Regression gating should use broad thresholds first, for example:

- fail on greater than 10 percent regression on stable smoke benchmarks,
- warn on greater than 5 percent regression,
- never gate on noisy exploratory suites.

## Deliverables

- Benchmark directory structure and harness entrypoints.
- JSON result schema and readers.
- Hardware and runtime metadata collection utilities.
- Benchmark spec files for existing MNIST, DQN, and tensor primitives.
- Baseline PyTorch runners for the first comparable workloads.
- CI smoke benchmark integration.
- Documentation for adding a new benchmark.

## Milestones

### Milestone A: Harness

- [x] Create benchmark directory structure.
- [x] Define benchmark manifest schema.
- [x] Add local CLI/build entrypoints.

### Milestone B: Initial Coverage

- [x] Primitive suite for representative tensor ops.
- [x] BLAS dot, matvec, and conv-lowering microbenchmarks with deterministic operands.
- [x] Autograd dot and matvec backward microbenchmarks.
- [x] Memory suite for cache high-water mark and graph arena reuse coverage.
- [x] Compiler suite for repeated eager graph-session capture on maintained
  reference models.
- [x] MNIST train and inference benchmarks using deterministic synthetic data.
- [x] Char-level language model train and inference benchmarks using deterministic synthetic causal windows.
- [x] DQN benchmark skeleton.
- [x] Synthetic GCN train and inference coverage.
- [x] Optional PyTorch baseline runner hook for comparable suites.

### Milestone C: Regression Policy

- [x] JSON result comparison utility.
- [x] Benchmark contract validator for committed specs and emitted JSONL artifacts.
- [x] CI smoke suite.
- [x] Backend-aware CUDA benchmark request smoke validation.
- [x] External baseline-runner contract smoke validation.
- [x] Published benchmark authoring guide.
- [x] Thread-sweep execution plus Markdown/JSON scaling reports for host runs.
- [x] Publication artifact smoke validation for comparison/provider/thread report outputs.

## Acceptance Criteria

- `zig build benchmark` runs from a clean checkout.
- At least three benchmark suites emit schema-valid JSON.
- Results include hardware and backend metadata.
- One baseline PyTorch comparison exists for a model benchmark.
- CI runs a smoke benchmark set and reports regression deltas.

## Risks

- Benchmarks can become noisy if thread placement and thermal state are not
  controlled.
- Framework comparisons can become unfair if preprocessing diverges.
- CI hardware variability can make strict thresholds brittle.

## Open Questions

- Should benchmark specs be JSON, Zig source, or both?
- Do we want per-operator flop accounting in the initial version?
- Should result archives live in the main repo or an external artifact store?

## Agentic Context

### 2026-03-28 Safetensors Checkpoint Interop Benchmark Suite

- Completed:
  - Extended
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig),
    [`benchmarks/src/validate.zig`](../../benchmarks/src/validate.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig), and
    [`build.zig`](../../build.zig)
    with a first-class `interop` suite plus the
    `zig build benchmark-interop` entrypoint, so RFC-0001 now benchmarks an
    executable import/export-facing surface instead of leaving interop coverage
    entirely to future ONNX/GGUF/ZML RFCs.
  - Added checked-in interop specs in
    [`benchmarks/specs/interop/`](../../benchmarks/specs/interop/)
    covering deterministic safetensors checkpoint export/import for MNIST MLP
    and CartPole-style DQN benchmark models, each measuring in-memory artifact
    translation and, for import, full model reconstruction from checkpoint
    bytes.
  - Updated
    [`tests/src/benchmark_smoke_main.zig`](../../tests/src/benchmark_smoke_main.zig),
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the interop suite is smoke-tested and documented alongside the existing
    primitive, compiler, model, memory, and CUDA-aware benchmark surfaces.
- Remains:
  - Extend the interop suite beyond safetensors checkpoints into ONNX import,
    GGUF load, and ZML translation cost once those RFCs produce executable
    artifact pipelines.
  - Decide whether interop records need dedicated tensor-count or artifact-size
    metadata in addition to the current checkpoint-byte throughput metric.
- Blockers:
  - This environment has no ONNX, GGUF, or ZML execution path yet, so the
    interop suite currently stops at maintained safetensors checkpoint
    encode/decode coverage rather than external-format translation benchmarks.
- Validation performed:
  - `zig build benchmark-interop`
  - `zig build benchmark-validate -- --group interop`
  - `zig build benchmark-validate -- --input benchmarks/results/interop.jsonl`
  - `zig build test-benchmark-smoke`
  - `zig build test`

### 2026-03-28 Compiler Capture Benchmark Suite

- Completed:
  - Extended
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig),
    [`benchmarks/src/validate.zig`](../../benchmarks/src/validate.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig), and
    [`build.zig`](../../build.zig)
    with a first-class `compiler` suite plus the
    `zig build benchmark-compiler` entrypoint, so RFC-0001 now benchmarks an
    executable compiler-facing surface instead of deferring all compiler
    coverage until the lazy-tensor RFC lands.
  - Added checked-in compiler capture specs in
    [`benchmarks/specs/compiler/`](../../benchmarks/specs/compiler/)
    covering MNIST MLP, char-LM, DQN, and GCN graph-capture workloads, each
    measuring forward-plus-loss graph construction on persistent model
    parameters with explicit teardown in the measured loop.
  - Extended
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py)
    so the new compiler kinds have an optional PyTorch baseline path instead of
    becoming Zig-only artifacts.
  - Updated
    [`tests/src/benchmark_smoke_main.zig`](../../tests/src/benchmark_smoke_main.zig),
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the compiler suite is smoke-tested and documented alongside the existing
    primitive, model, memory, and CUDA-aware benchmark surfaces.
- Remains:
  - Extend the compiler suite beyond eager capture into optimization-pass
    timing and realized execution once RFC-0006 and RFC-0007 land executable
    graph pipelines.
  - Decide whether compiler capture rows need dedicated node-count or pass
    telemetry beyond the current graph-arena and allocator memory fields.
- Blockers:
  - This environment has no lazy-tensor or optimizer-pass pipeline yet, so the
    compiler suite currently stops at eager graph capture plus teardown rather
    than true optimization or lowering benchmarks.
- Validation performed:
  - `zig build test`
  - `zig build test-benchmark-smoke`
  - `zig build test-benchmark-baseline-smoke`
  - `zig build benchmark-compiler`
  - `zig build benchmark-validate -- --group compiler`
  - `zig build benchmark -- --spec benchmarks/specs/compiler/mnist-mlp-capture-synthetic.json --baseline pytorch --output .zig-cache/compiler-mnist-capture-baseline.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/compiler-mnist-capture-baseline.jsonl`

### 2026-03-28 Char-LM Benchmark Coverage

- Completed:
  - Extended
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig), and
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py)
    with `char_lm_train` and `char_lm_infer`, so RFC-0001 now benchmarks a
    maintained language-model family alongside MNIST, DQN, and GCN.
  - Added checked-in specs in
    [`benchmarks/specs/model-train/char-lm-synthetic.json`](../../benchmarks/specs/model-train/char-lm-synthetic.json)
    and
    [`benchmarks/specs/model-infer/char-lm-synthetic.json`](../../benchmarks/specs/model-infer/char-lm-synthetic.json),
    plus smoke coverage through
    [`tests/src/benchmark_smoke_main.zig`](../../tests/src/benchmark_smoke_main.zig).
  - Updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the new language-model benchmark surface is documented next to the
    existing reference-model workflows.
- Remains:
  - Publish first cross-framework char-LM benchmark artifacts on a host with a
    working PyTorch baseline environment.
  - Expand from MLP-style causal windows to a transformer-style workload once
    RFC-0012 grows a stronger sequence-model operator surface.
- Blockers:
  - No verified local PyTorch baseline artifact was captured in this run, so
    the new benchmark kinds were validated through Zig smoke execution and the
    existing baseline-runner contract path rather than published cross-runner
    measurements.
- Validation performed:
  - `zig build test-example-smoke`
  - `zig build test-benchmark-smoke`

### 2026-03-28 Pendulum Dynamics Benchmark Coverage

- Completed:
  - Extended
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig), and
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py)
    with `pendulum_dynamics_train` and `pendulum_dynamics_infer`, so RFC-0001
    now benchmarks a maintained physics/control workload alongside MNIST,
    char-LM, DQN, and GCN.
  - Added checked-in specs in
    [`benchmarks/specs/model-train/pendulum-dynamics-synthetic.json`](../../benchmarks/specs/model-train/pendulum-dynamics-synthetic.json)
    and
    [`benchmarks/specs/model-infer/pendulum-dynamics-synthetic.json`](../../benchmarks/specs/model-infer/pendulum-dynamics-synthetic.json),
    plus smoke coverage through
    [`tests/src/benchmark_smoke_main.zig`](../../tests/src/benchmark_smoke_main.zig).
  - Updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the new physics/control benchmark surface is documented next to the
    existing reference-model workflows.
- Remains:
  - Add compiler-capture coverage for the pendulum family once RFC-0006 and
    RFC-0007 expose a meaningful captured-graph optimization path for this
    workload.
  - Capture and publish broader pendulum benchmark result sets once dedicated
    publication runs are scheduled.
- Blockers:
  - CUDA-capable validation remains unavailable in this environment, so the
    new benchmark slice was validated on host only.
- Validation performed:
  - `zig build test-benchmark-smoke`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/pendulum-dynamics-synthetic.json --baseline pytorch --output .zig-cache/pendulum-dynamics-baseline.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/pendulum-dynamics-baseline.jsonl`

### 2026-03-28 Backend-Aware CUDA Benchmark Requests

- Completed:
  - Extended
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/result.zig`](../../benchmarks/src/result.zig),
    [`benchmarks/src/metadata.zig`](../../benchmarks/src/metadata.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig),
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig), and
    [`benchmarks/src/validate.zig`](../../benchmarks/src/validate.zig)
    so checked-in benchmark specs can declare a target device, CUDA-targeted
    Zig runs emit explicit `skipped` or `failed` records instead of aborting
    the harness, and successful CUDA runs carry structured `backend.cuda`
    metadata in the JSONL contract.
  - Added checked-in CUDA-targeted model specs in
    [`benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json`](../../benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json)
    and
    [`benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json`](../../benchmarks/specs/model-train/dqn-cartpole-synthetic-cuda.json)
    so RFC-0001 now has concrete backend-aware benchmark surfaces for model
    inference and training.
  - Added
    [`tests/src/benchmark_cuda_request_smoke_main.zig`](../../tests/src/benchmark_cuda_request_smoke_main.zig)
    plus `zig build test-benchmark-cuda-request-smoke` in
    [`build.zig`](../../build.zig),
    and updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md),
    so the backend-aware contract and CUDA degradation path are documented and
    smoke-tested end to end.
  - Extended the external PyTorch baseline contract so CUDA-targeted specs now
    emit explicit skipped baseline rows instead of silently omitting the
    non-Zig runner record.
- Remains:
  - Execute the new CUDA-targeted specs on a real toolkit/device host and
    publish the first non-skipped CUDA benchmark artifacts.
  - Extend the same device-aware benchmark contract to future compiler and
    interop suites as those workloads become executable.
- Blockers:
  - This environment still has no CUDA toolkit or CUDA device available, so
    the new RFC-0001 CUDA benchmark slice validated through explicit skip
    semantics and JSONL contract checks rather than real accelerator timings.
- Validation performed:
  - `zig build test-benchmark-cuda-request-smoke`
  - `zig build test-benchmark-smoke`
  - `zig build test-benchmark-baseline-smoke`
  - `zig build test-benchmark-publication-smoke`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json --output .zig-cache/benchmark-cuda-spec.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/benchmark-cuda-spec.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic-cuda.json --baseline pytorch --output .zig-cache/benchmark-cuda-pytorch.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/benchmark-cuda-pytorch.jsonl`
  - `zig build test`

### 2026-03-28 Benchmark Publication Bundle

- Completed:
  - Added
    [`benchmarks/src/publication_bundle.zig`](../../benchmarks/src/publication_bundle.zig)
    and
    [`benchmarks/src/publication_bundle_main.zig`](../../benchmarks/src/publication_bundle_main.zig),
    plus the `zig build benchmark-publication-bundle` entrypoint in
    [`build.zig`](../../build.zig),
    so RFC-0001 can now package candidate, baseline, and auxiliary JSONL
    artifacts alongside compare/provider/thread reports into a single
    validated JSON manifest plus Markdown summary.
  - Extended
    [`tests/src/benchmark_publication_smoke_main.zig`](../../tests/src/benchmark_publication_smoke_main.zig)
    so publication smoke coverage now includes the bundle manifest/summary in
    addition to compare/provider/thread outputs.
  - Updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md),
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md), and
    [`.github/workflows/benchmark-smoke.yml`](../../.github/workflows/benchmark-smoke.yml)
    so the CI smoke path now emits a real thread sweep and publishes the bundle
    manifest/summary beside the raw JSONL and report artifacts.
- Remains:
  - Extend the same publication-bundle contract to future CUDA, compiler, and
    interop suites once those artifact types exist.
  - Decide whether downstream docs/reporting automation should ingest the
    bundle manifest directly instead of pointing at individual files.
- Blockers:
  - This environment still lacks OpenBLAS/oneMKL runtime access, so the new
    bundle contract validates publication structure and thread-scaling outputs
    locally but cannot yet package real cross-provider provider-report inputs.
- Validation performed:
  - `zig build test-benchmark-publication-smoke`
  - `zig build benchmark-publication-bundle -- --help`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output .zig-cache/publication-bundle-candidate.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json --thread-count 1 --thread-count 2 --output .zig-cache/publication-thread-sweep.jsonl`
  - `zig build benchmark-thread-report -- --input .zig-cache/publication-thread-sweep.jsonl --baseline-thread-count 1 --markdown-output .zig-cache/publication-thread-report.md --json-output .zig-cache/publication-thread-report.json`
  - `zig build benchmark-publication-bundle -- --candidate-jsonl .zig-cache/publication-bundle-candidate.jsonl --extra-results-jsonl .zig-cache/publication-thread-sweep.jsonl --thread-report-json .zig-cache/publication-thread-report.json --thread-report-markdown .zig-cache/publication-thread-report.md --manifest-output .zig-cache/publication-manifest.json --summary-output .zig-cache/publication-summary.md`
  - `zig build test`

### 2026-03-28 Baseline Runner Contract Smoke

- Completed:
  - Hardened
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig) so requested
    PyTorch baseline runs no longer fail open: launcher errors, non-zero exits,
    empty output, and malformed JSONL now emit structured `failed` baseline
    records with the original spec metadata instead of silently omitting the
    baseline row.
  - Added
    [`tests/src/benchmark_baseline_smoke_main.zig`](../../tests/src/benchmark_baseline_smoke_main.zig)
    plus the stdlib-only fixture runner
    [`tests/fixtures/benchmark_baseline_smoke_runner.py`](../../tests/fixtures/benchmark_baseline_smoke_runner.py)
    and wired `zig build test-benchmark-baseline-smoke` through
    [`build.zig`](../../build.zig) so RFC-0001 now smoke-tests successful
    external baseline emission, malformed JSONL fallback, and missing-runner
    fallback end to end.
  - Updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the benchmark contract documents the new baseline failure semantics and
    the dedicated smoke gate.
- Remains:
  - Capture non-skipped PyTorch baseline data on a machine with `torch`
    installed and fold those results into published comparison artifacts.
  - Extend the same contract-hardening approach to future non-PyTorch external
    baseline runners if the harness grows beyond the current Python path.
- Blockers:
  - This environment still has no local `torch` installation, so the real
    PyTorch path validated through explicit `skipped` output while the new smoke
    gate used a stdlib fixture runner to verify success and failure semantics.
- Validation performed:
  - `python3 -m py_compile tests/fixtures/benchmark_baseline_smoke_runner.py benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build test-benchmark-baseline-smoke`
  - `zig build benchmark -- --spec benchmarks/specs/model-infer/mnist-mlp-synthetic.json --baseline pytorch --output .zig-cache/pytorch-baseline-smoke.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/pytorch-baseline-smoke.jsonl`
  - `zig build test-benchmark-smoke`
  - `zig build test-benchmark-publication-smoke`
  - `zig build test`

### 2026-03-28 Benchmark Publication Artifact Smoke

- Completed:
  - Added
    [`tests/src/benchmark_publication_smoke_main.zig`](../../tests/src/benchmark_publication_smoke_main.zig)
    and the `zig build test-benchmark-publication-smoke` entrypoint in
    [`build.zig`](../../build.zig)
    so RFC-0001 now validates compare/provider/thread publication artifacts in
    addition to raw benchmark JSONL contract checks.
  - The smoke flow executes a real thread-swept primitive benchmark, validates
    the emitted JSONL artifact, derives schema-faithful candidate and
    alternate-provider variants, and then writes plus re-parses comparison,
    provider-report, and thread-report outputs to catch publication regressions
    before they reach CI or docs workflows.
  - Updated
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md)
    so the new smoke gate is documented alongside the existing validator and
    reporting steps.
- Remains:
  - Replace the synthetic alternate-provider smoke input with real OpenBLAS and
    oneMKL runs when multi-provider runners become available.
  - Mirror the same publication artifact smoke coverage for future CUDA,
    compiler, and interop report surfaces.
- Blockers:
  - The current machine still only exposes the Accelerate host provider, so
    the provider-report smoke slice validates report generation using a
    schema-faithful synthetic OpenBLAS variant instead of a real second
    provider run.
- Validation performed:
  - `zig build test-benchmark-publication-smoke`
  - `zig build test`

### 2026-03-28 Benchmark Contract Validator

- Completed:
  - Added
    [`benchmarks/src/validate.zig`](../../benchmarks/src/validate.zig),
    [`benchmarks/src/validate_main.zig`](../../benchmarks/src/validate_main.zig),
    and the `zig build benchmark-validate` entrypoint in
    [`build.zig`](../../build.zig)
    so committed benchmark specs and emitted JSONL artifacts can be validated
    directly against the RFC-0001 contract.
  - The validator cross-checks JSONL records against the referenced checked-in
    spec, enforces summary-stat and metadata invariants, and rejects duplicate
    result identities within a single artifact before downstream comparison or
    publication steps consume the output.
  - Added
    [`tests/src/benchmark_smoke_main.zig`](../../tests/src/benchmark_smoke_main.zig)
    and `zig build test-benchmark-smoke`, which executes one representative
    checked-in spec per suite through the real benchmark harness and then runs
    the validator on the emitted artifact.
  - Updated benchmark documentation in
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md),
    and [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md).
- Remains:
  - Extend the validator as CUDA, compiler, and interop suites add new
    accelerator-specific fields and benchmark identities.
  - Decide whether CI should archive validator JSON/text reports alongside the
    existing comparison outputs once publication workflows mature.
- Blockers:
  - None for the host benchmark contract slice; this run validated completely
    on the local Accelerate host.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-validate`
  - `zig build test-benchmark-smoke`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output .zig-cache/zigrad-benchmark-validate.jsonl`
  - `zig build benchmark-validate -- --input .zig-cache/zigrad-benchmark-validate.jsonl`

### 2026-03-28 Benchmark Provenance Contract

- Completed:
  - Promoted explicit benchmark provenance into the checked-in spec contract by
    requiring `provenance.data_source` and `provenance.preprocessing` in
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig) and in
    every committed JSON spec under [`benchmarks/specs/`](../../benchmarks/specs/).
  - Extended the JSONL result schema in
    [`benchmarks/src/result.zig`](../../benchmarks/src/result.zig),
    [`benchmarks/src/metadata.zig`](../../benchmarks/src/metadata.zig), and
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig) so every emitted
    record now carries the originating `spec_path`, declared provenance, CPU
    frequency policy when discoverable, and captured host thread-environment
    hints.
  - Mirrored the same reproducibility metadata in the optional PyTorch baseline
    runner at
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py)
    so cross-framework JSONL comparisons remain schema-aligned.
  - Updated benchmark documentation in
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md).
- Remains:
  - Extend the same provenance discipline to future dataset-backed suites so
    dataset manifests and preprocessing revisions are encoded as explicitly as
    the current synthetic workloads.
  - Carry equivalent accelerator-specific environment metadata once CUDA and
    compiler benchmark suites become executable.
- Blockers:
  - CPU frequency policy is only discoverable on Linux hosts that expose the
    standard cpufreq governor file; this macOS validation run therefore
    exercised the field as optional metadata rather than a populated output.
- Validation performed:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json --output /tmp/zigrad-benchmark-provenance.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/blas/dot-f32-262144.json --baseline pytorch --thread-count 2 --output /tmp/zigrad-benchmark-baseline-provenance.jsonl`

### 2026-03-28 Host Thread Scaling Workflow

- Completed:
  - Added repeatable `--thread-count <n>` overrides to
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig) so a single
    checked-in benchmark spec can execute as a deterministic thread sweep
    without cloning JSON files.
  - Updated the optional PyTorch baseline shim in
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py)
    so baseline records honor the same overridden thread count metadata and
    runtime thread setting as the Zig harness.
  - Added a dedicated host thread-scaling report generator in
    [`benchmarks/src/thread_report.zig`](../../benchmarks/src/thread_report.zig)
    with the matching
    [`benchmarks/src/thread_report_main.zig`](../../benchmarks/src/thread_report_main.zig)
    entrypoint and `zig build benchmark-thread-report` build step.
  - Updated
    [`benchmarks/src/compare.zig`](../../benchmarks/src/compare.zig)
    so regression comparisons key records by benchmark id, runner, and thread
    count; sweep JSONL files now compare cleanly instead of colliding on
    duplicate ids.
  - Documented the workflow in
    [`README.md`](../../README.md),
    [`benchmarks/README.md`](../../benchmarks/README.md), and
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md).
- Remains:
  - Collect published OpenBLAS and oneMKL thread-scaling runs for the same
    benchmark groups.
  - Extend the same sweep/report surface to future CUDA and compiler benchmark
    suites as those RFCs become executable.
- Blockers:
  - This run only exercised the local Accelerate backend, so cross-provider
    scaling data still remains unpublished.
- Validation performed:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark-thread-report -- --help`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json --thread-count 1 --thread-count 2 --output /tmp/zigrad-thread-sweep.jsonl`
  - `zig build benchmark-compare -- --baseline /tmp/zigrad-thread-sweep.jsonl --candidate /tmp/zigrad-thread-sweep.jsonl --runner zig`
  - `zig build benchmark-thread-report -- --input /tmp/zigrad-thread-sweep.jsonl --baseline-thread-count 1 --markdown-output /tmp/zigrad-thread-scaling.md --json-output /tmp/zigrad-thread-scaling.json`

### 2026-03-28 Host Dispatch Telemetry Promotion

- Completed:
  - Extended [`benchmarks/src/result.zig`](../../benchmarks/src/result.zig)
    so Zig benchmark records can carry `backend.host_blas_telemetry`,
    including low-level BLAS call counts plus direct-batched versus
    fallback-broadcast dispatch counts.
  - Updated [`.gitignore`](../../.gitignore) so
    [`benchmarks/specs/`](../../benchmarks/specs/) JSON files are no longer
    hidden behind the repository-wide `*.json` ignore rule; the benchmark
    harness inputs are now committed product surface.
  - Threaded the telemetry through
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig),
    [`benchmarks/src/cli.zig`](../../benchmarks/src/cli.zig), and
    [`benchmarks/src/metadata.zig`](../../benchmarks/src/metadata.zig), with
    counters reset after warmup so the metadata reflects measured work rather
    than setup or warmup noise.
  - Added a deterministic nested-broadcast primitive matmul spec under
    [`benchmarks/specs/primitive/`](../../benchmarks/specs/primitive/) so the
    smoke suite exercises the manual fallback path as part of RFC-0001.
  - Updated benchmark docs in
    [`benchmarks/README.md`](../../benchmarks/README.md),
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md), and
    [`README.md`](../../README.md).
- Remains:
  - Add benchmark-visible telemetry for CUDA/compiler/interop suites as those
    RFCs become executable.
  - Capture non-skipped PyTorch baseline data on a machine with `torch`
    installed.
- Blockers:
  - This run still validated only the macOS Accelerate path, so OpenBLAS and
    oneMKL benchmark records carrying the new telemetry remain unexecuted.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-primitive -- --output /tmp/zigrad-primitive.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/primitive/matmul-broadcast-fallback-f32-2x2x2x3-2x1x3x2.json --output /tmp/zigrad-broadcast-fallback.jsonl`
  - `zig build benchmark-models -- --output /tmp/zigrad-models.jsonl`

### 2026-03-28 Conv Lowering Coverage

- Completed:
  - Added a reusable legacy conv helper in
    [`src/nn/conv_utils.zig`](../../src/nn/conv_utils.zig) that lowers square
    conv2d inputs through `im2col` plus batched matmul and reshapes the output
    back into NCHW form.
  - Extended the benchmark manifest and workload runner so the `blas` suite now
    supports `blas_conv2d_im2col` specs with `stride`, `padding`, and
    `dilation` fields.
  - Added two deterministic conv-lowering benchmark specs under
    [`benchmarks/specs/blas/`](../../benchmarks/specs/blas/) and expanded the
    optional PyTorch baseline runner to emit comparable records for the new
    workload kind.
  - Added correctness and benchmark execution regression tests for the new
    workload plus provider telemetry coverage for the legacy conv path.
- Remains:
  - Add CUDA-targeted suites and compiler/interop coverage as the dependent
    RFCs become executable.
  - Gather PyTorch baseline data on a machine with `torch` installed instead of
    relying on the explicit skip-path validation.
- Blockers:
  - No local `torch` install was available during this run, so the conv
    baseline path validated only the emitted `skipped` record rather than an
    executed framework comparison.
- Validation performed:
  - `zig build test`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --output /tmp/zigrad-conv-benchmark.jsonl`
  - `zig build benchmark-blas -- --output /tmp/zigrad-blas.jsonl`
  - `zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json --baseline pytorch --output /tmp/zigrad-conv-benchmark-baseline.jsonl`

### 2026-03-27

- Completed:
  - Added the benchmark harness under [`benchmarks/`](../benchmarks/) with
    manifest parsing, deterministic workload execution, JSON-lines result
    emission, metadata capture, and `zig build benchmark*` entrypoints.
  - Added initial benchmark specs for primitive add/matmul and synthetic
    MNIST-style MLP train/infer workloads.
  - Added an optional PyTorch baseline runner that emits compatible `skipped`
    records when `torch` is unavailable.
  - Added CI smoke execution in
    [`.github/workflows/benchmark-smoke.yml`](../../.github/workflows/benchmark-smoke.yml).
  - Added the benchmark comparison utility under
    [`benchmarks/src/compare.zig`](../../benchmarks/src/compare.zig) with
    threshold-based pass/warn/fail classification, JSON and text reports, and a
    `zig build benchmark-compare` entrypoint.
  - Added the benchmark authoring guide in
    [`benchmarks/AUTHORING.md`](../../benchmarks/AUTHORING.md).
  - Updated benchmark smoke CI to compare the current checkout against the base
    revision on the same runner and upload comparison artifacts.
- Remains:
  - Expand benchmark coverage further into CUDA, compiler, and interop suites.
  - Add richer reporting and historical storage beyond per-run JSON/text
    artifacts.
- Blockers:
  - No local `torch` install was available during this run, so the PyTorch
    runner was only validated through its explicit skip path.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-compare -- --help`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/latest.jsonl --candidate benchmarks/results/latest.jsonl --runner zig --json-output benchmarks/results/comparison.json --report-output benchmarks/results/comparison.txt`
  - `zig build benchmark-primitive`
  - `zig build benchmark-models`
  - `ruby -e 'require "yaml"; YAML.load_file(".github/workflows/benchmark-smoke.yml"); puts "workflow ok"'`
  - `python3 -c "import json, pathlib; print(pathlib.Path('benchmarks/results/comparison.json').exists())"`
- Exact commands:
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

### 2026-03-27 Synthetic DQN + GCN Coverage

- Completed:
  - Added synthetic CartPole-style DQN train/infer benchmark kinds and specs.
  - Added synthetic two-layer GCN train/infer benchmark kinds and specs with a
    deterministic ring-plus-skip graph topology.
  - Expanded the optional PyTorch runner to understand DQN and GCN specs while
    preserving explicit `skipped` behavior when `torch` is unavailable.
  - Added benchmark tests covering DQN infer and GCN train execution paths.
- Remains:
  - Exercise the expanded PyTorch runner on a machine with `torch` installed.
  - Add additional suites for CUDA, compiler, and interop milestones.
- Blockers:
  - Local validation still lacked a `torch` installation, so baseline execution
    validated only the explicit skip records.
- Validation performed:
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`

### 2026-03-27 Memory Suite Coverage

- Completed:
  - Added allocator telemetry to the host caching allocator so benchmark runs
    can report peak live bytes, final live bytes, and peak scratch bytes.
  - Added graph arena capacity reporting and a new `memory` benchmark suite
    with host tensor cache cycle and synthetic MNIST training-step workloads.
  - Extended benchmark records and comparison reports to carry memory metrics
    and flag memory regressions alongside latency regressions.
- Remains:
  - Add CUDA memory telemetry and device-transfer accounting once RFC-0003
    moves deeper into supported-path work.
  - Add interop and compiler memory suites when those RFCs land executable
    workloads.
- Blockers:
  - Memory telemetry currently reflects Zigrad-native host allocator and graph
    behavior only; there is no comparable PyTorch baseline for these specs yet.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-memory`
  - `zig build benchmark`
  - `zig build benchmark-compare -- --baseline benchmarks/results/memory.jsonl --candidate benchmarks/results/memory.jsonl --runner zig --json-output benchmarks/results/memory-comparison.json --report-output benchmarks/results/memory-comparison.txt`
  - `zig build test`
  - `zig build benchmark-models`
  - `zig build benchmark`
  - `zig build benchmark-models -- --baseline pytorch`

### 2026-03-27 BLAS + Autograd Linear Algebra Coverage

- Completed:
  - Added dedicated `blas` benchmark kinds for deterministic dot and matvec
    workloads plus JSON specs under
    [`benchmarks/specs/blas/`](../../benchmarks/specs/blas/).
  - Added dedicated `autograd` benchmark kinds for deterministic dot and
    matvec forward+backward workloads plus JSON specs under
    [`benchmarks/specs/autograd/`](../../benchmarks/specs/autograd/).
  - Extended the manifest parser, CLI group loading, and `zig build`
    entrypoints to support the new suites.
  - Implemented the Zig workloads and benchmark tests for BLAS forward and
    autograd backward execution paths.
  - Expanded the optional PyTorch runner so prepared machines can emit baseline
    records for BLAS and autograd suites in addition to the model suites.
- Remains:
  - Add CUDA-specific suites once RFC-0003 begins landing measurable kernels.
  - Add compiler, interop, and memory suites to cover the rest of the RFC-0001
    taxonomy.
- Blockers:
  - Local validation still lacked a `torch` installation, so PyTorch execution
    validated the explicit skip path and Python compilation rather than live
    parity timings.
- Validation performed:
  - `zig build test`
  - `zig build benchmark-blas`
  - `zig build benchmark-autograd`
  - `zig build benchmark`
  - `zig build benchmark -- --baseline pytorch --group blas`
  - `zig build benchmark -- --baseline pytorch --group autograd`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
- Exact commands:
  - `zig build test`
  - `zig build benchmark-blas`
  - `zig build benchmark-autograd`
  - `zig build benchmark`
  - `zig build benchmark -- --baseline pytorch --group blas`
  - `zig build benchmark -- --baseline pytorch --group autograd`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
