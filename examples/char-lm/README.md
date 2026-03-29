# Zigrad Char-Level Language Model

This reference example trains a small causal self-attention next-character
model on an embedded corpus. It is intentionally lightweight: no tokenizer
download, no dataset fetch, and no Python dependency.

## What it validates

- sequence-style model wiring in eager Zigrad
- causal attention plus residual next-token readout on top of one-hot inputs
- reproducible next-token batching from a fixed corpus
- smoke-testable training and greedy generation
- RFC-0001 benchmark integration for model-train and model-infer paths

## Backend expectation

This example uses the shared runtime-device selector. `ZG_DEVICE=host` is the
default, and `ZG_DEVICE=cuda[:index]` is supported when the example is built
with `-Denable_cuda=true`.

## Dataset and artifacts

- Corpus source: embedded text in [`src/corpus.txt`](./src/corpus.txt)
- Preprocessing: character vocabulary discovery in first-seen order, fixed
  causal windows, one-hot context tensors, one-hot next-token labels
- Checkpoint path: `char-lm.stz` by default, storing token/position embeddings,
  attention projections, and the output head

## Run

```sh
zig build run
```

Override the prompt used for greedy generation:

```sh
ZG_CHAR_LM_PROMPT="graph " zig build run
```

Run the fast smoke configuration:

```sh
ZG_EXAMPLE_SMOKE=1 zig build run
```

Optional CUDA smoke when the example is built with CUDA support:

```sh
ZG_DEVICE=cuda ZG_EXAMPLE_SMOKE=1 zig build run -Denable_cuda=true
```
