# Zigrad Pendulum Dynamics

This reference example trains a small MLP to predict the next state of a
deterministic torque-driven pendulum from periodic state features
`(sin(theta), cos(theta), omega, torque)`. It ships with no external simulator
dependency and runs from a clean checkout.

## What it validates

- a maintained physics/control-oriented example in the RFC-0012 portfolio
- deterministic synthetic dynamics generation with explicit normalization
- smoke-testable regression training plus rollout evaluation
- RFC-0001 benchmark hooks for model-train and model-infer paths

## Backend expectation

This example uses the shared runtime-device selector. `ZG_DEVICE=host` is the
default, and `ZG_DEVICE=cuda[:index]` is supported when the example is built
with `-Denable_cuda=true`.

## Dataset and artifacts

- Transition source: generated analytically from an in-repo pendulum step
  function in [`src/dataset.zig`](./src/dataset.zig)
- Preprocessing: encode wrapped angle state as sine/cosine features, normalize
  angular velocity and torque, and predict sine/cosine next-state targets plus
  normalized next angular velocity
- Checkpoint path: `pendulum.stz` by default

## Run

```sh
zig build run
```

Run the fast smoke configuration:

```sh
ZG_EXAMPLE_SMOKE=1 zig build run
```

Optional CUDA smoke when the example is built with CUDA support:

```sh
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```
