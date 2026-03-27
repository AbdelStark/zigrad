# gcn

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
