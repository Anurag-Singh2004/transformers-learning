# Mini-GPT — A Decoder-Only Transformer Built From Scratch

A small (~835K parameter) GPT-style language model, implemented from scratch in PyTorch : every component (tokenization, embeddings, causal multi-head attention, feed-forward layers, residual connections, layer normalization, and sampling strategies) was built and understood individually before being assembled here.

Trained on real Python source code, this model learns to generate syntactically plausible (if imperfect) Python-like text purely through next-character prediction — no pretrained weights, no external libraries beyond PyTorch's core building blocks.

## Architecture

- **Tokenization**: character-level (not BPE) — deliberate choice, since exact whitespace/indentation matters for Python syntax, and code has a naturally small character vocabulary
- **Vocabulary size**: 101 unique characters
- **Model**: 4 stacked transformer blocks, 4 attention heads, d_model=128, d_ff=512
- **Positional encoding**: learned (not sinusoidal) — same approach GPT-2 uses
- **Total parameters**: ~835,000

## Dataset

Combined source code from two well-known, MIT-licensed Python libraries:
- [`requests`](https://github.com/psf/requests)
- [`click`](https://github.com/pallets/click)

~653KB of real, idiomatic Python code. Used purely for educational/training purposes.

## Training

Trained for 3,000 steps on CPU. Loss decreased from **4.90 → 1.42**.

Sample generations at increasing training steps show a clear qualitative progression — from pure random symbols (step 0) to recognizable Python patterns: `def` declarations, `self.` attribute access, type hints, docstring-style comments, and reasonably consistent indentation (by step ~1500+).

## Sampling strategies

Implemented temperature, top-k, and top-p (nucleus) sampling from scratch, applied to the trained model's output distribution. Lower temperature produces more conservative, syntactically coherent output; higher temperature produces more varied but less structured text.

## KV Caching Benchmark

Implemented KV caching (reusing previously computed Key/Value projections
instead of recomputing them at every generation step) and benchmarked it
against naive generation.

Result: **1.61x speedup** generating 100 tokens (0.44s naive vs 0.27s cached).

At this small model/sequence scale, caching overhead limits the speedup,
but the mechanism is correctly demonstrated — at production scale (longer
sequences, larger models), this same technique provides dramatically
larger speedups, which is why it's a standard requirement for real LLM
inference serving.

## Known limitations

- **No end-of-sequence handling**: the vocabulary has no `<eos>` token, so generation always runs for a fixed number of characters rather than naturally stopping at a logical boundary. A future improvement would be inserting an EOS marker between training files and teaching the model to predict it.
- **Small scale**: at ~835K parameters and 653KB of training data, this model demonstrates the transformer *mechanism* working correctly — it does not produce reliably correct, runnable Python code, and isn't intended to.
- **Character-level ceiling**: character-level tokenization means the model must learn spelling/syntax entirely from scratch, character by character, which is a harder learning problem than subword-level tokenization would present.

## Running it

```bash
pip install torch
python train.py    # trains from scratch, saves mini_gpt_weights.pt
python generate.py  # generates text with configurable sampling
python trace_through.py # step-by-step trace of data flow through the model
```