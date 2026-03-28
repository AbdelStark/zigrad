# gcn

## Backend expectation

This reference example uses the shared runtime-device selector. `ZG_DEVICE=host`
is the default, and `ZG_DEVICE=cuda[:index]` is supported when the example is
built with `-Denable_cuda=true`.

The message-passing and mask/evaluation paths now avoid host-only tensor reads,
although dedicated CUDA hardware validation is still pending.

## download dataset

```bash
uv run ref/dataset.py
```

## train

```bash
# pytorch
uv run ref/train.py

# zigrad
zig build run -Doptimize=ReleaseFast
# if you have oneMKL installed:
zig build run -Doptimize=ReleaseFast -Dhost_blas=mkl
```
