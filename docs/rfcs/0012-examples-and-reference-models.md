# RFC-0012: Examples and Reference Models Program

Status: `Planned`  
Priority: `P1`  
Depends on: RFC-0001, RFC-0002, RFC-0003  
Blocks: RFC-0005  
Last updated: `2026-03-27`

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

