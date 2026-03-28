# gcn

## Backend expectation

This reference example is currently host-only. The message-passing path still
contains host-view assumptions, so `ZG_DEVICE=cuda` is rejected intentionally
until the remaining device-safety work is completed.

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
