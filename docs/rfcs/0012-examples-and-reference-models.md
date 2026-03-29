# RFC-0012: Examples and Reference Models Program

Status: `Ready`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0002, RFC-0003  
Blocks: RFC-0005  
Last updated: `2026-03-29`

## Summary

This RFC defines the example and reference-model program used to validate the
entire roadmap. It turns examples from ad hoc demos into maintained product
artifacts with quality bars, benchmark hooks, dependency expectations, and CI
smoke coverage.

## Motivation

The roadmap explicitly calls out more examples such as LLMs, physics, and
robotic control. Without a program-level plan, examples tend to drift, bit-rot,
or fail to exercise the most important product surfaces. Zigrad needs a curated
portfolio that simultaneously serves:

- users learning the framework,
- developers validating new backends,
- performance and compiler benchmarking,
- interoperability testing.

## Goals

- Define a stable example portfolio across domains.
- Establish a quality bar for example inclusion and maintenance.
- Tie examples to benchmark and CI coverage.
- Use examples to validate real roadmap capabilities.
- Keep datasets and dependencies explicit and manageable.

## Non-Goals

- Shipping every interesting model immediately.
- Supporting heavyweight optional dependencies in the default build path.
- Treating examples as production product packages.
- Replacing documentation prose with examples alone.

## Reference Portfolio

The first portfolio should include:

- `cv`: MNIST and at least one more modern vision example,
- `rl`: DQN plus a cleaner control/reference task,
- `graph`: GCN or another graph workload,
- `llm`: small transformer or GPT-style inference/training reference,
- `physics-control`: one deterministic simulation or robotic-control example.

## Quality Bar

An example qualifies as a reference example only if it has:

- a clear README,
- pinned or reproducible data/model inputs,
- a known-good command line,
- smoke-test coverage,
- benchmark integration where relevant,
- declared hardware/backend expectations,
- explicit optional dependency instructions if not default-safe.

## Dependency Policy

Examples with heavyweight dependencies must:

- be clearly marked optional,
- fail gracefully when dependencies are missing,
- avoid contaminating default library builds,
- provide setup instructions that match current build tooling.

## Dataset and Artifact Management

Each example must define:

- dataset source,
- preprocessing steps,
- caching location,
- checkpoint expectations,
- whether artifacts are committed, downloaded, or generated.

For large models, manifest-driven download or conversion steps should be
documented rather than hidden in bespoke scripts.

## Work Breakdown

### Workstream A: Portfolio Curation

- define which examples are canonical,
- retire or demote examples that do not meet the quality bar.

### Workstream B: Example Hardening

- build and run scripts,
- error messages,
- dependency docs,
- smoke tests.

### Workstream C: Benchmark Integration

- tie examples into RFC-0001 benchmark suites,
- record backend and hardware metadata,
- publish expected throughput/latency reference ranges.

### Workstream D: New Roadmap Examples

- [x] LLM example, initial char-level causal language model reference with
  embedded corpus, smoke coverage, and benchmark hooks.
- [x] physics/robotics example.
- [x] upgraded RL example.

## Testing Plan

- build tests for every maintained example,
- runtime smoke tests for a curated subset,
- benchmark coverage for reference models,
- artifact validation for downloadable checkpoints or datasets.

## Acceptance Criteria

- The project has a documented reference example portfolio.
- Every reference example has a README and reproducible entrypoint.
- CI smoke coverage exists for the portfolio subset that is practical.
- At least one LLM, one RL/control, and one physics/robotics-oriented example
  are maintained as reference examples.

## Risks

- Example maintenance burden can grow quickly with optional dependencies.
- Large-model examples can dominate CI and local setup costs.
- Example quality may lag feature work unless ownership is explicit.

## Open Questions

- Which LLM example is small enough to maintain but still meaningful?
- Should Mujoco live inside the main repo, or remain an optional external setup?
- How much artifact caching should the repo standardize?

## Agentic Context

### 2026-03-28 Char-Level Language Model Reference Example

- Completed:
  - Added
    [`examples/char-lm/src/model.zig`](../../examples/char-lm/src/model.zig),
    [`examples/char-lm/src/dataset.zig`](../../examples/char-lm/src/dataset.zig),
    [`examples/char-lm/src/main.zig`](../../examples/char-lm/src/main.zig),
    and the embedded corpus
    [`examples/char-lm/src/corpus.txt`](../../examples/char-lm/src/corpus.txt),
    giving RFC-0012 its first maintained `llm` reference example that trains,
    evaluates, and greedily generates text from a clean checkout.
  - Added standalone build/docs in
    [`examples/char-lm/build.zig`](../../examples/char-lm/build.zig),
    [`examples/char-lm/build.zig.zon`](../../examples/char-lm/build.zig.zon),
    and
    [`examples/char-lm/README.md`](../../examples/char-lm/README.md),
    then wired the example into repo-level smoke coverage through
    [`build.zig`](../../build.zig)
    and
    [`tests/src/example_smoke_main.zig`](../../tests/src/example_smoke_main.zig).
  - Extended RFC-0001 benchmark coverage with char-LM model-train/model-infer
    specs so the new reference family participates in the maintained benchmark
    surface immediately instead of living only as a demo.
- Remains:
  - Revisit the `llm` portfolio with a transformer-style model once the
    sequence-model primitive surface grows beyond one-hot MLP baselines.
- Blockers:
  - CUDA-capable validation remains pending, so the new language-model example
    was verified on host plus shared runtime-device wiring rather than actual
    accelerator execution.
- Validation performed:
  - `zig build test-example-smoke`
  - `zig build test-benchmark-smoke`

### 2026-03-29 Causal Self-Attention Char-LM Upgrade

- Completed:
  - Replaced the maintained char-level language model’s flattened affine stack
    with a causal self-attention architecture in
    [`examples/char-lm/src/model.zig`](../../examples/char-lm/src/model.zig)
    and
    [`examples/char-lm/src/main.zig`](../../examples/char-lm/src/main.zig),
    including token/position embeddings, causal masking, residual mixing, and
    a last-token readout that still trains and greedily generates from the
    embedded corpus in smoke mode.
  - Updated RFC-0001 benchmark and interop integration through
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig)
    and
    [`benchmarks/runners/pytorch/mnist_mlp.py`](../../benchmarks/runners/pytorch/mnist_mlp.py),
    so the maintained char-LM train, infer, compiler-capture, and
    safetensors import/export slices all measure the same attention-based
    parameter layout instead of the old four-tensor affine checkpoint.
  - Refreshed the shipped docs in
    [`examples/char-lm/README.md`](../../examples/char-lm/README.md),
    [`README.md`](../../README.md), and
    [`benchmarks/README.md`](../../benchmarks/README.md)
    so the reference portfolio now describes the landed attention model
    instead of the old MLP-style baseline.
- Remains:
  - Decide when RFC-0012 should grow from this single-block attention baseline
    into a deeper or tokenized transformer reference model.
  - Validate the same workload on CUDA-capable hardware once that environment
    is available.
- Blockers:
  - No CUDA-capable runtime or real PyTorch baseline environment was available
    in this run, so the new model was validated through Zig smoke paths and a
    Python syntax check rather than executed cross-runner benchmark artifacts.
- Validation performed:
  - `zig build test-example-smoke`
  - `zig build test-benchmark-smoke`
  - `python3 -m py_compile benchmarks/runners/pytorch/mnist_mlp.py`
  - `zig build test`

### 2026-03-28 Pendulum Dynamics Reference Example

- Completed:
  - Added
    [`examples/pendulum/src/dataset.zig`](../../examples/pendulum/src/dataset.zig),
    [`examples/pendulum/src/model.zig`](../../examples/pendulum/src/model.zig),
    [`examples/pendulum/src/main.zig`](../../examples/pendulum/src/main.zig),
    [`examples/pendulum/build.zig`](../../examples/pendulum/build.zig),
    [`examples/pendulum/build.zig.zon`](../../examples/pendulum/build.zig.zon),
    and
    [`examples/pendulum/README.md`](../../examples/pendulum/README.md),
    giving RFC-0012 its first maintained physics/control reference example
    with deterministic transition generation, regression training, and rollout
    evaluation from a clean checkout.
  - Wired the example into repo-level smoke coverage through
    [`build.zig`](../../build.zig)
    and
    [`tests/src/example_smoke_main.zig`](../../tests/src/example_smoke_main.zig),
    including a rollout-RMSE regression threshold so the new example has a
    concrete maintained quality bar instead of a compile-only check.
  - Extended RFC-0001 benchmark coverage with pendulum
    model-train/model-infer specs, so the maintained physics/control family is
    exercised through the same benchmark harness as the rest of the reference
    portfolio.
  - RFC-0012 now satisfies its acceptance criterion requiring at least one
    maintained `llm`, one RL/control example, and one physics/robotics-style
    example.
- Remains:
  - Revisit the pendulum family with richer control or model-based-planning
    workflows once the examples program grows beyond single-step dynamics
    regression.
- Blockers:
  - CUDA-capable validation remains unavailable in this environment, so the
    new example was validated on host only even though it uses the shared
    runtime-device selector.
- Validation performed:
  - `zig build test-example-smoke`
  - `cd examples/pendulum && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `zig build test-benchmark-smoke`
  - `zig build test`

### 2026-03-28 Corridor Control Reference Example

- Completed:
  - Added
    [`examples/corridor/src/environment.zig`](../../examples/corridor/src/environment.zig),
    [`examples/corridor/src/replay_buffer.zig`](../../examples/corridor/src/replay_buffer.zig),
    [`examples/corridor/src/model.zig`](../../examples/corridor/src/model.zig),
    [`examples/corridor/src/main.zig`](../../examples/corridor/src/main.zig),
    [`examples/corridor/build.zig`](../../examples/corridor/build.zig),
    [`examples/corridor/build.zig.zon`](../../examples/corridor/build.zig.zon),
    and
    [`examples/corridor/README.md`](../../examples/corridor/README.md),
    giving RFC-0012 a maintained upgraded RL/reference-control example with a
    deterministic momentum-constrained environment, replay-buffer training,
    checkpoint support, and greedy evaluation from a clean checkout.
  - Wired the example into repo-level smoke coverage through
    [`build.zig`](../../build.zig)
    and
    [`tests/src/example_smoke_main.zig`](../../tests/src/example_smoke_main.zig),
    including an evaluation-improvement gate so the example must actually
    learn instead of merely compiling.
  - Extended RFC-0001 benchmark coverage through
    [`benchmarks/src/manifest.zig`](../../benchmarks/src/manifest.zig),
    [`benchmarks/src/workload.zig`](../../benchmarks/src/workload.zig),
    [`benchmarks/specs/model-train/corridor-control-synthetic.json`](../../benchmarks/specs/model-train/corridor-control-synthetic.json),
    and
    [`benchmarks/specs/model-infer/corridor-control-synthetic.json`](../../benchmarks/specs/model-infer/corridor-control-synthetic.json),
    so the upgraded RL slice participates in the maintained benchmark harness
    instead of living only as a standalone example.
  - Completed RFC-0012 Workstream D's remaining upgraded RL example item.
- Remains:
  - Decide whether the next RL/reference step should be actor-critic or a
    richer continuous-control workload once the maintained portfolio needs a
    harder backend/compiler stress case.
- Blockers:
  - CUDA-capable validation remains unavailable in this environment, so the
    new reference-control example and benchmark slice were verified on host
    only.
- Validation performed:
  - `cd examples/corridor && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `zig build test-example-smoke`
  - `zig build test-benchmark-smoke`

### 2026-03-28 Device-Safe Losses For Maintained Training Examples

- Completed:
  - Reworked
    [`src/nn/loss.zig`](../../src/nn/loss.zig)
    so the shared maintained-example loss surface
    (`softmax_cross_entropy_loss`, `softmax`, `smooth_l1_loss`, and
    `mse_loss`) keeps its direct host implementation but stages tensors through
    explicit host copies on off-host devices instead of reading device memory
    directly from Zig.
  - Added a regression test in
    [`src/nn/tests/test_loss.zig`](../../src/nn/tests/test_loss.zig)
    for `softmax` over a non-last dimension, tightening coverage around the
    stride-aware host reference implementation the maintained examples now
    depend on.
  - Revalidated the maintained smoke portfolio through `zig build test`, which
    continues to exercise hello-world, MNIST, DQN, and GCN after the loss
    changes.
- Remains:
  - Run the same maintained training paths on a real CUDA-capable host now
    that the core loss surface no longer assumes host-readable tensors.
  - Continue expanding RFC-0012 with new reference examples once the current
    maintained portfolio has sustained backend validation.
- Blockers:
  - No CUDA toolkit or CUDA device was available in this run, so the updated
    loss surface validated through host smoke coverage and unit tests rather
    than actual accelerator execution.
- Validation performed:
  - `zig build test`

### 2026-03-28 Device-Safe Adam For Maintained Training Examples

- Completed:
  - Fixed
    [`src/nn/optim.zig`](../../src/nn/optim.zig)
    so Adam advances its timestep once per logical optimizer step instead of
    once per attached parameter, restoring correct bias correction for the
    multi-parameter DQN and GCN training examples.
  - Added a backend-dispatched Adam update path across
    [`src/device/opspec.zig`](../../src/device/opspec.zig),
    [`src/device/host_device.zig`](../../src/device/host_device.zig),
    [`src/device/cuda_device.zig`](../../src/device/cuda_device.zig),
    and the new CUDA kernel
    [`src/cuda/blas/adam.cu`](../../src/cuda/blas/adam.cu),
    so the maintained training examples no longer depend on a host-only
    optimizer implementation.
  - Added optimizer coverage in
    [`src/nn/optim.zig`](../../src/nn/optim.zig)
    and revalidated the maintained smoke portfolio through the top-level test
    entrypoint.
- Remains:
  - Exercise the same DQN and GCN training paths on a real CUDA-capable host.
  - Keep expanding the portfolio with new reference examples after the current
    set has sustained backend validation.
- Blockers:
  - No CUDA toolkit or CUDA device was available in this run, so the example
    portfolio validated through host smoke coverage and shared optimizer tests
    rather than actual GPU execution.
- Validation performed:
  - `zig build test`

### 2026-03-28 DQN + GCN CUDA Example Enablement

- Completed:
  - Promoted the DQN and GCN reference entrypoints to the shared runtime-device
    contract: both now accept `ZG_DEVICE=host|cpu|cuda[:index]` when built with
    `-Denable_cuda=true`.
  - Removed the concrete device-safety blockers in the example code:
    DQN’s gather path now stages offsets correctly for device memory, and GCN’s
    masking plus evaluation code now performs explicit host copies instead of
    reading CUDA-backed tensor storage directly.
  - Updated the synthetic GCN smoke dataset to use a non-prefix train mask so
    the maintained smoke path exercises the corrected masked forward/backward
    mapping instead of only the contiguous-mask case.
  - Updated the top-level and example README surfaces so the documented backend
    expectations match the shipped runtime behavior.
- Remains:
  - Run dedicated CUDA smoke coverage for DQN and GCN on a GPU-capable runner.
  - Expand the portfolio with new reference examples from later RFC-0012
    workstreams once the current set has sustained backend validation.
- Blockers:
  - No CUDA hardware was available in this run, so example validation covered
    host execution plus code-path audit rather than actual GPU execution.
- Validation performed:
  - `zig build test`
  - `cd examples/dqn && ZG_EXAMPLE_SMOKE=1 zig build run`
  - `cd examples/gcn && ZG_EXAMPLE_SMOKE=1 zig build run`

### 2026-03-28 Explicit Example Backend Expectations

- Completed:
  - Moved the maintained example entrypoints onto the shared runtime-device
    selector from [`src/device/runtime_device.zig`](../../src/device/runtime_device.zig),
    so backend intent now lives in code instead of ad hoc commented-out device
    setup.
  - Marked current backend expectations explicitly across the maintained
    examples and documented the runtime selector in the README surfaces.
  - Updated the standalone example build scripts and README surfaces so the
    repo now documents the same backend contract that the code enforces.
  - Removed a host-storage assumption from MNIST evaluation by adding the
    `NDTensor.to_host_owned(...)` helper in [`src/ndtensor.zig`](../../src/ndtensor.zig).
- Remains:
  - Add dedicated README coverage for the MNIST and hello-world runtime device
    contract if those examples grow beyond the top-level README guidance.
- Blockers:
  - No CUDA hardware was available in this run, so the newly declared backend
    expectations were validated on host plus explicit error behavior only.
- Validation performed:
  - `zig build test`
  - `cd examples/hello-world && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
  - `cd examples/dqn && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
  - `cd examples/gcn && zig build --help | rg "enable_cuda|rebuild_cuda|host_blas"`
