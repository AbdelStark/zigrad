# AGENTS.md

This file defines how coding agents should operate in this repository.

It is intentionally strict. Favor correctness, reproducibility, explicit state,
and small validated increments over speed-by-guessing.

## Mission

Zigrad is a research-first ML framework that must remain:

- fast,
- inspectable,
- reproducible,
- backend-aware,
- pleasant for iterative model development.

Agents working here should preserve the eager-first user experience while
systematically building toward the roadmap in [docs/roadmap.md](./docs/roadmap.md).

## Canonical Planning Docs

Treat these as the source of truth for roadmap work:

- [Roadmap Index](./docs/roadmap.md)
- [RFC-0001 Benchmarking Program](./docs/rfcs/0001-benchmarking-program.md)
- [RFC-0002 oneMKL Host Backend](./docs/rfcs/0002-onemkl-host-backend.md)
- [RFC-0003 CUDA Backend](./docs/rfcs/0003-cuda-backend.md)
- [RFC-0012 Examples and Reference Models](./docs/rfcs/0012-examples-and-reference-models.md)
- [RFC-0006 Lazy Tensors](./docs/rfcs/0006-lazy-tensors.md)
- [RFC-0007 Static Graph Optimization](./docs/rfcs/0007-static-graph-optimization.md)
- [RFC-0004 ONNX Interop](./docs/rfcs/0004-onnx-interop.md)
- [RFC-0005 ggml/GGUF Interop](./docs/rfcs/0005-ggml-gguf-interop.md)
- [RFC-0010 ZML Inference Bridge](./docs/rfcs/0010-zml-inference-bridge.md)
- [RFC-0008 Dynamic Graph Compiler](./docs/rfcs/0008-dynamic-graph-compiler.md)
- [RFC-0009 MLIR Lowering Pipeline](./docs/rfcs/0009-mlir-lowering-pipeline.md)
- [RFC-0011 Apache TVM Integration](./docs/rfcs/0011-apache-tvm-integration.md)

## Priority Order

Unless the user explicitly redirects the work, choose tasks in this order:

1. RFC-0001 Standardized Benchmarking Program
2. RFC-0002 oneMKL Host Backend
3. RFC-0003 CUDA Backend
4. RFC-0012 Examples and Reference Models
5. RFC-0006 Lazy Tensors
6. RFC-0007 Static Graph Optimization
7. RFC-0004 ONNX Interop
8. RFC-0005 ggml/GGUF Interop
9. RFC-0010 ZML Inference Bridge
10. RFC-0008 Dynamic Graph Compiler
11. RFC-0009 MLIR Lowering Pipeline
12. RFC-0011 Apache TVM Integration

When selecting work:

- prefer `Ready` RFCs before `Planned` RFCs,
- prefer unblocked work before blocked work,
- prefer thin vertical slices that produce validated progress,
- do not start a lower-priority RFC if a higher-priority RFC has obvious,
  unblocked, high-value work remaining.

## Current Execution State (2026-03-30)

Read this section first to understand where the project stands before selecting
work. For detailed milestone specs, see
[`docs/next-milestones.md`](./docs/next-milestones.md).

**Phase 0 (Measurement/Backend): Substantially complete.**
RFC-0001, RFC-0002, RFC-0003 have comprehensive landed implementations. The
main remaining work is real-GPU CUDA validation (needs hardware) and published
provider comparison runs. No further agent work needed unless directed.

**Phase 1 (Execution Model): Active frontier.**
- RFC-0006 has observe-mode capture AND deferred forward execution landed.
  Next: deferred backward pass (M-6).
- RFC-0007 has Graph IR, verifier, pass manager, DCE, execution bridge,
  **constant folding**, **algebraic simplification**, and **CSE** landed.
  All milestone passes complete. Transpose/layout simplification remains as extension.
- RFC-0012 has all reference examples landed with smoke + benchmark coverage.
  Next: deeper transformer portfolio (lower priority).
- RFC-0004 (ONNX) has import MVP landed (protobuf parser, schema, op registry,
  importModel/importGraph lowering). Export and full opset remain.
- RFC-0005 (GGUF) has reader MVP landed (parser, dequantizer, loader).
  Next: additional quantized formats (Milestone B), example integration (Milestone C).

**Phase 2 (Compilation): Dependencies landed, not yet started.**
RFC-0008 (Dynamic Compiler) has both RFC-0006 and RFC-0007 available now. It
needs a scoping spike before implementation. RFC-0009 (MLIR) is blocked on
RFC-0008.

**Phase 3 (External Compilers): Not started, low priority.**

**Where to start:** The highest-value unblocked work is **deferred backward
(M-6)**, which completes the lazy tensor training story. ONNX export (M-4b)
can follow the import MVP. GGUF Milestone B/C are lower priority extensions.
RFC-0008 (Dynamic Compiler) now has all prerequisite passes landed and is
ready for a scoping spike.

## Core Agent Rules

- Do not guess about behavior that can be verified from the codebase.
- Do not silently change user-facing semantics without updating the relevant RFC.
- Do not hide build, benchmark, or environment assumptions.
- Do not add optional dependency coupling to the default path unless the RFC
  explicitly requires it.
- Do not claim performance wins without benchmark evidence.
- Do not land broad refactors without a narrowed objective, explicit validation,
  and updated docs.
- Do not leave the repo in a less reproducible state than you found it.

## Required Task Loop

For every substantial task, follow this loop:

1. Read the relevant RFC and any directly dependent RFCs.
2. Identify the smallest implementable slice that advances the RFC.
3. Inspect the current code paths, build logic, tests, and examples affected.
4. State assumptions explicitly in your working notes or status update.
5. Implement the slice with minimal unrelated churn.
6. Verify with the narrowest meaningful tests first, then broader integration
   checks as needed.
7. Update roadmap/RFC status and agentic context files before finishing.
8. Summarize what changed, what was verified, and what remains.

If a task cannot be validated locally, say so explicitly and record the missing
validation in the status update.

## Required Status Updates After Each Task

After each completed task, update the planning docs so the next agent has
accurate context. This is mandatory.

At minimum:

- update the relevant RFC `Status` if the task changes readiness or execution
  state,
- add or update a short progress note in the relevant RFC milestone/checklist
  section when useful,
- update [docs/roadmap.md](./docs/roadmap.md) if the roadmap matrix, dependency
  picture, or recommended order changed,
- add or update an "agentic context" section in the touched RFC or roadmap doc
  with:
  - what was completed,
  - what remains,
  - blockers,
  - validation performed,
  - exact commands used where they matter.

If the repository later adopts a dedicated progress log such as
`docs/agentic-context.md`, use that as the canonical rolling context file and
link it from the touched RFCs. Until then, keep the context close to the RFC.

## Coding Standards

- Preserve the eager-first API unless the RFC explicitly changes it.
- Prefer small, composable abstractions over speculative framework-wide layers.
- Keep backend-specific logic isolated behind explicit boundaries.
- Prefer clear dataflow and ownership over clever indirection.
- Treat allocator behavior, memory lifetime, and cleanup paths as first-class.
- Add comments only where they remove real ambiguity.
- Keep docs and implementation terminology aligned.

## Verification Standards

Every change must be verified at an appropriate level:

- unit tests for local logic,
- integration tests for cross-module behavior,
- example smoke tests for user-facing workflows,
- benchmarks for any performance-related claim,
- CPU and CUDA parity checks where backend behavior is touched.

Minimum rule: no code change is complete until the most relevant local build or
test command has been run, unless impossible in the environment.

## World-Class Harness Practices

Agents must treat benchmarking and testing infrastructure as product surface.

- Make commands deterministic.
- Pin seeds where randomness exists.
- Record environment details when they affect results.
- Separate warmup from measured iterations.
- Distinguish compile/setup cost from steady-state execution cost.
- Avoid hidden caches and undocumented external state.
- Emit machine-readable outputs for benchmark results where practical.
- Keep smoke coverage fast and broad coverage schedulable.
- Never merge a benchmark shortcut that makes published claims less trustworthy.

When touching benchmarked code, consult
[RFC-0001](./docs/rfcs/0001-benchmarking-program.md) and keep result reporting
compatible with it.

## Build and Dependency Practices

- Keep optional integrations optional.
- Fail clearly when a dependency is missing.
- Do not hard-code machine-local paths in committed files.
- Keep vendored third-party code attributed, licensed, and documented.
- Prefer hermetic or near-hermetic execution paths when feasible.
- If a build step depends on external tooling, document the exact requirement.

## Documentation Practices

When behavior, architecture, or sequencing changes:

- update the relevant RFC,
- update [docs/roadmap.md](./docs/roadmap.md) if scope or priority changed,
- update README only for user-visible behavior or entrypoint changes,
- keep checked-in markdown links repository-relative.

## Commit Practices

- Keep commits focused.
- Use commit messages that describe the actual change, not just the symptom.
- Do not mix unrelated cleanup into roadmap work without a strong reason.
- Include docs updates in the same commit when they are part of the change.

## Escalation Rules

Stop and ask for direction when:

- two RFCs conflict materially,
- the next step requires a policy decision not captured in the RFCs,
- a change would break eager-first semantics or default build ergonomics,
- the required validation environment is unavailable and the risk is high.

Otherwise, make the smallest reasonable assumption, state it, and continue.
