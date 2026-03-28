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
3. Add a JSON spec under [`benchmarks/specs/`](./specs/). If the workload is
   meant to verify a provider-sensitive path, choose shapes that make the
   expected `backend.host_blas_telemetry` signal obvious.
4. Add or update an optional baseline runner when parity exists.
5. Add tests for parsing, execution logic, and any comparison edge cases.

## Spec Rules

Specs are JSON files grouped by suite under [`benchmarks/specs/`](./specs/).
They are part of the committed benchmark contract, not ignored local outputs.

Required fields:

- `id`
- `suite`
- `kind`
- `dtype`
- `warmup_iterations`
- `measured_iterations`
- `seed`
- `provenance.data_source`
- `provenance.preprocessing`

Shape and batch fields depend on the workload kind. The manifest validator
should reject incomplete specs with clear errors. Batched workloads should make
their leading dimension explicit through `batch_size`; graph workloads may omit
`batch_size` when node count is carried by `input_shape[0]`.

For conv-lowering benchmarks, encode the input tensor in `lhs_shape`, the
weights in `rhs_shape`, and the lowering parameters in `stride`, `padding`, and
`dilation`.

Treat provenance as part of the benchmark contract, not optional commentary:

- `data_source` should identify where the benchmark inputs come from.
- `preprocessing` should list the deterministic shaping/materialization steps
  that turn the source into benchmark-ready tensors.
- Keep provenance concise but specific enough that a later result archive can be
  understood without opening the workload implementation.

## Validation Workflow

Run the narrowest meaningful validation first:

```sh
zig build test
zig build benchmark -- --spec benchmarks/specs/primitive/add-f32-1024x1024.json
zig build benchmark -- --spec benchmarks/specs/blas/dot-f32-262144.json
zig build benchmark -- --spec benchmarks/specs/blas/conv2d-im2col-f32-batch4-1x28x28-k3-out8.json
zig build benchmark -- --spec benchmarks/specs/primitive/matmul-f32-256x256x256.json --thread-count 1 --thread-count 2
```

For smoke-scope changes, rerun the standard entrypoints:

```sh
zig build benchmark-primitive
zig build benchmark-blas
zig build benchmark-autograd
zig build benchmark-memory
zig build benchmark-models
zig build benchmark
```

If the benchmark exercises a specific host provider configuration, record that
explicitly in the command, for example `zig build benchmark -Dhost_blas=openblas`
or `zig build benchmark -Dhost_blas=mkl`.

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

Thread-sweep outputs are first-class inputs to `benchmark-compare`. Records are
matched by benchmark id, runner, and configured thread count, so repeated
thread-count rows can be compared without collapsing into duplicate ids.

## Host Provider Reports

RFC-0002 provider benchmarking now has a dedicated reporting step for
publishable host BLAS tables. After running the same benchmark group under each
provider, generate a consolidated report:

```sh
zig build benchmark -Dhost_blas=accelerate -- --group blas --output benchmarks/results/accelerate-blas.jsonl
zig build benchmark -Dhost_blas=openblas -- --group blas --output benchmarks/results/openblas-blas.jsonl
zig build benchmark-provider-report -- \
  --input benchmarks/results/accelerate-blas.jsonl \
  --input benchmarks/results/openblas-blas.jsonl \
  --baseline-provider accelerate \
  --markdown-output benchmarks/results/host-provider-blas.md \
  --json-output benchmarks/results/host-provider-blas.json
```

Guidelines:

- Keep thread counts aligned across providers so the grouped rows stay directly
  comparable.
- Preserve benchmark ids across provider runs; the report groups on benchmark
  id plus thread count.
- Use `--runner zig` unless you intentionally want to inspect another runner's
  host records.
- Treat the Markdown output as publishable artifact and the JSON output as
  machine-readable input for docs or CI post-processing.

## Thread Scaling Reports

Use repeated `--thread-count` overrides when you need to study scaling without
duplicating checked-in specs:

```sh
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

Guidelines:

- Keep the benchmark id stable across the sweep; thread count lives in backend
  metadata and is part of report/comparison grouping.
- Use the same provider across the sweep when the goal is scaling efficiency.
- Pick a baseline thread count that is present in every group if you need
  uniform cross-benchmark comparisons; otherwise let the report choose the
  smallest successful thread count per group.
- Treat scaling efficiency as a diagnostic, not a claim. Publish the raw
  latency and speedup columns alongside it.

## CI Expectations

The smoke workflow benchmarks the current checkout and a baseline revision on
the same runner, then compares the resulting JSONL files with
`benchmark-compare`.

When changing smoke-covered specs:

- preserve benchmark ids when semantics stay compatible,
- update the RFC and roadmap context if ids must change,
- keep runtime bounded enough for CI,
- ensure the comparison output remains readable and actionable.
