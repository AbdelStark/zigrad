# Zigrad Reinforcement Learning - Cartpole DQN

Example implementation of DQN to solve the cartpole task. Extremely fast with a torch implementation for comparison. Includes a PyTorch implementation which uses python bindings to the cartpole environment written in Zig.

https://github.com/user-attachments/assets/16c838d9-6e31-46be-9ba3-c73487652a43

The above rendering is a sneak peek into a Zigrad + [Raylib](https://www.raylib.com) demo not yet released (video is not using a Zigrad trained model).

## Backend expectation

This example uses the shared runtime-device selector. `ZG_DEVICE=host` is the
default, and `ZG_DEVICE=cuda[:index]` is supported when the example is built
with `-Denable_cuda=true`.

The replay buffer intentionally remains host-resident, but sampled batches,
policy evaluation, and gather/backprop paths no longer require host-backed
tensor storage.

## Tensorboard integration

Metrics are logged to tensorboard with Zigrad's `TensorboardLogger`

![](tensorboard-dqn.png)
