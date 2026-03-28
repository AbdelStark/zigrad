# RFC-0012: Examples and Reference Models Program

Status: `Planned`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0002, RFC-0003  
Blocks: RFC-0005  
Last updated: `2026-03-28`

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

- LLM example,
- physics/robotics example,
- upgraded RL example.

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
