# Benchmark Authoring Guide

This guide describes the RFC-0001 workflow for adding or updating benchmarks in
[`benchmarks/`](./).

## Invariants

- Keep the default path reproducible from a clean checkout.
- Prefer deterministic synthetic inputs unless a dataset manifest is part of the
  benchmark contract.
- For graph or RL-style workloads, keep synthetic topology and transition
  generation deterministic and documented in the workload implementation.
- Record stable `benchmark_id` values. Renaming an existing smoke benchmark
  breaks historical comparisons and should be intentional.
- Separate setup cost from steady-state timing.
- Do not add new mandatory dependencies to the default harness path.

## Benchmark Types

Current suites and kinds are defined in
[`benchmarks/src/manifest.zig`](./src/manifest.zig)
and executed in
[`benchmarks/src/workload.zig`](./src/workload.zig).

When adding a benchmark:

1. Add or extend the manifest enum and validation rules.
2. Implement the workload in `workload.zig`.
3. Add a JSON spec under [`benchmarks/specs/`](./specs/).
4. Add or update an optional baseline runner when parity exists.
5. Add tests for parsing, execution logic, and any comparison edge cases.

## Spec Rules

Specs are JSON files grouped by suite under [`benchmarks/specs/`](./specs/).

Required fields:

- `id`
- `suite`
- `kind`
- `dtype`
- `warmup_iterations`
- `measured_iterations`
- `seed`

Shape and batch fields depend on the workload kind. The manifest validator
should reject incomplete specs with clear errors. Batched workloads should make
their leading dimension explicit through `batch_size`; graph workloads may omit
`batch_size` when node count is carried by `input_shape[0]`.

## Validation Workflow

Run the narrowest meaningful validation first:

```sh
zig build test
zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json
```

For smoke-scope changes, rerun the standard entrypoints:

```sh
zig build benchmark-primitive
zig build benchmark-models
zig build benchmark
```

## Regression Comparison

Use the comparison tool to compare a candidate run against a baseline JSONL:

```sh
zig build benchmark-compare -- \
  --baseline benchmarks/results/baseline.jsonl \
  --candidate benchmarks/results/latest.jsonl \
  --runner zig \
  --json-output benchmarks/results/comparison.json \
  --report-output benchmarks/results/comparison.txt
```

Default thresholds follow RFC-0001:

- warn above 5% mean-latency regression
- fail above 10% mean-latency regression

Added benchmark records are reported but do not fail the comparison. Missing
baseline-covered records do fail the comparison because they break regression
continuity for smoke suites.

## CI Expectations

The smoke workflow benchmarks the current checkout and a baseline revision on
the same runner, then compares the resulting JSONL files with
`benchmark-compare`.

When changing smoke-covered specs:

- preserve benchmark ids when semantics stay compatible,
- update the RFC and roadmap context if ids must change,
- keep runtime bounded enough for CI,
- ensure the comparison output remains readable and actionable.
