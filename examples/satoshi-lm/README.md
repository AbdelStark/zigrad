# Satoshi Language Model

A character-level language model trained on Satoshi Nakamoto's known writings
(code comments, whitepaper excerpts, forum posts, and emails) using Zigrad's
eager autodiff engine.

The goal is to learn the statistical patterns of Satoshi's writing style at the
character level and generate new text that resembles his distinctive voice:
precise technical explanations, understated tone, and clear reasoning about
cryptographic protocols and distributed systems.

## What it demonstrates

- Causal self-attention on a real-world corpus (~470KB of Satoshi's text)
- Train/validation split with per-epoch loss monitoring
- Temperature-controlled text generation (greedy to creative)
- Stride-based dataset subsampling for practical training times
- Checkpoint save/load for incremental training

## Dataset

The corpus (`src/satoshi-all-text.txt`) contains all known Satoshi Nakamoto
writings: Bitcoin source code comments, the whitepaper, Bitcointalk forum posts,
and cryptography mailing list emails.

- **Source**: [where-is-satoshi](https://github.com/basvandorst/where-is-satoshi/blob/main/data/_satoshi/all-text.txt) by Bas van Dorst
- ~470KB, ~5000 lines
- 90/10 train/validation split (at nearest line boundary)
- Character-level vocabulary (~98 unique bytes)
- Embedded at compile time via `@embedFile`

## Architecture

Single-layer causal self-attention transformer (`SatoshiLmModel`) with:
- Token + positional embeddings
- Q/K/V projections with causal masking
- Residual connection + last-token readout
- Linear output head over vocabulary

Default: `context_len=64`, `hidden_size=128` (~78K parameters).

This is a tiny model by modern standards -- the point is to show end-to-end
training and generation in pure Zig with Zigrad, not to produce production
quality text. With enough epochs the model learns recognizable word fragments,
common Satoshi phrases ("proof-of-work", "transaction", "block chain"), and
the cadence of his forum-post style.

## Run

```sh
zig build run
```

Override the generation prompt:

```sh
ZG_SATOSHI_LM_PROMPT="Bitcoin is " zig build run
```

Fast smoke test:

```sh
ZG_EXAMPLE_SMOKE=1 zig build run
```

With CUDA:

```sh
ZG_DEVICE=cuda zig build run -Denable_cuda=true
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Samples per training batch |
| `num_epochs` | 5 | Training epochs |
| `context_len` | 64 | Character context window |
| `hidden_size` | 128 | Embedding / attention dimension |
| `stride` | 16 | Step between consecutive samples |
| `temperature` | 0.8 | Generation temperature (0 = greedy) |
| `learning_rate` | 0.001 | Adam learning rate |
| `val_split` | 0.1 | Fraction of corpus for validation |

## Tips

- **Longer training**: Increase `num_epochs` or decrease `stride` (more samples
  per epoch) for better results. Lower `stride` means more overlap between
  training windows.
- **More coherent output**: Lower `temperature` (e.g. 0.3) produces more
  predictable text; higher values (e.g. 1.2) produce more creative but noisier
  output.
- **Resume training**: The model auto-loads from `satoshi-lm.stz` if it exists,
  so running multiple times continues from where you left off.
