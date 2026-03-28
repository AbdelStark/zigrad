# Corridor Control Reference Example

This example adds a deterministic RL/control task to the maintained reference
portfolio. It trains a small Q-network with replay and a target network on a
momentum-constrained corridor environment:

- actions: `left`, `coast`, `right`
- state features: normalized position, velocity, and distance-to-goal
- rewards: shaped progress, terminal goal bonus, and pit penalty
- environment: deterministic and embedded, so it runs from a clean checkout

## Run

```sh
cd examples/corridor
zig build run
```

Smoke mode keeps the run short while still validating end-to-end training and
greedy evaluation:

```sh
cd examples/corridor
ZG_EXAMPLE_SMOKE=1 zig build run
```

## Backend Expectation

This example uses the shared runtime-device selector. `ZG_DEVICE=host` is the
default, and `ZG_DEVICE=cuda[:index]` is supported when the example is built
with `-Denable_cuda=true`.

The environment and replay buffer stay host-resident. Sampled batches, Q-value
evaluation, target computation, and optimizer updates use the selected tensor
backend.

Optional CUDA smoke run:

```sh
cd examples/corridor
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```

## Checkpoints

By default the example loads from and saves to `corridor.stz` in the example
directory. Smoke mode disables checkpoints so the runtime stays hermetic.
