# Zigrad Reinforcement Learning - Cartpole DQN

Example implementation of DQN to solve the cartpole task. Extremely fast with a torch implementation for comparison. Includes a PyTorch implementation which uses python bindings to the cartpole environment written in Zig.

https://github.com/user-attachments/assets/16c838d9-6e31-46be-9ba3-c73487652a43

The above rendering is a sneak peek into a Zigrad + [Raylib](https://www.raylib.com) demo not yet released (video is not using a Zigrad trained model).

## Backend expectation

This example now uses the shared runtime-device selector, but it should still
be treated as host-only for now. `ZG_DEVICE=host` is the default; requesting
CUDA fails intentionally until the remaining replay-buffer and validation work
for RFC-0003 lands.

## Tensorboard integration

Metrics are logged to tensorboard with Zigrad's `TensorboardLogger`

![](tensorboard-dqn.png)
