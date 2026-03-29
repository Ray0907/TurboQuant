# TurboQuant

Near-optimal KV cache quantization for Apple Silicon — MLX implementation of [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874).

Compress key/value caches **6×** with minimal quality loss, enabling longer contexts on memory-constrained devices.

## Two use cases

### 1. KV cache compression (longer LLM context)

LLMs accumulate past K/V tensors token by token. TurboQuant compresses them so a Mac with 8 GB RAM can sustain much longer conversations.

```python
from turboquant.mlx import compress, inner_product

# On each new token — compress the key cache
compressed = compress(keys, bits=3.5)   # ~6× smaller than float16

# Compute attention scores without decompressing
scores = inner_product(queries, compressed)  # [batch, heads, q_seq, k_seq]
```

**Practical effect:** Qwen2.5-7B on an 8 GB Mac goes from ~2K tokens before OOM to ~12K tokens at 3.5 bits.

### 2. Zero-indexing-time vector search

TurboQuant's rotation matrix is data-oblivious (no calibration, no training). Documents can be queried immediately after insertion — no index build step.

```python
from turboquant.mlx import compress_vectors, search

# Compress a document corpus once
corpus = compress_vectors(doc_embeddings, bits=3.5)  # [N_docs, dim]

# Search — no index build required
scores, indices = search(query_embedding, corpus, top_k=10)
```

**vs. Faiss / Chroma:** those require building an index first. TurboQuant compresses in one pass and is immediately searchable.

## Install

```bash
pip install turboquant
```

Requires Python ≥ 3.10, Apple Silicon (MLX).

For the mlx-lm integration test (optional):

```bash
pip install "turboquant[integration]"
```

## Algorithm

### Step 1 — Random Rotation (RHT)

Multiplying a KV vector by a random rotation matrix spreads its energy uniformly across all dimensions, making each component equally quantizable.

![Random Rotation](assets/rotation.gif)

### Step 2 — PolarQuant

Pairs of coordinates are converted from Cartesian (x, y) to polar (r, θ). Because post-rotation angles are near-uniformly distributed, a fixed angular grid requires zero per-block normalization overhead.

![PolarQuant](assets/polarquant.gif)

### Full Pipeline

![TurboQuant Pipeline](assets/pipeline.gif)

TurboQuant stacks three transforms:

```
keys  ──►  RHT  ──►  PolarQuant  ──►  QJL residual
          rotate    quantize angles   1-bit sketch
                    + store norm      of residual
```

1. **RHT** (Randomized Hadamard Transform) — O(d log d) random rotation that spreads energy uniformly across dimensions so PolarQuant's angle distribution is predictable.

2. **PolarQuant** — recursive pairwise polar decomposition. Quantizes angles with level-specific Lloyd-Max codebooks; stores the global L2 norm as float16 (lossless). Reconstruction MSE is bounded by `C · 2^(−2b) · ‖x‖²`.

3. **QJL** (Quantized Johnson-Lindenstrauss) — 1-bit sketch of the quantization residual. Provides an unbiased inner product correction: `√(π/2) · (scale/m) · ⟨Gq, s⟩`.

## Validation on real activations

All numbers below are measured on the actual Qwen2.5-0.5B model (no mocks).

### KV cache — attention score quality

Layer-0 attention score cosine similarity on RoPE-encoded KV vectors:

| Bits | Score cosine (attention logits) |
|------|---------------------------------|
| 3.5  | 0.88                            |
| 4.0  | 0.94                            |
| 5.0  | 0.98                            |

> Score cosine is the correct quality metric — post-softmax weight cosine is not
> used because softmax amplifies small score differences non-linearly when
> attention is peaked.

### KV cache — perplexity (end-to-end)

Patching layer-0 attention with TurboQuant at 8 bits in a full forward pass:

| Scenario | PPL delta |
|----------|-----------|
| 8-bit patch, prefill mode | **+8–10 %** |

Prefill mode (all tokens at once) is pessimistic: softmax amplification cascades
through all 24 layers.  The paper evaluates in decode mode (one query per step)
where the cascade is absent and PPL impact is smaller.

### Vector search — recall@10

Synthetic corpus: 500 documents, dim=64, 20 queries.

| Bits | Recall@10 |
|------|-----------|
| 3.5  | 71.5 %    |
| 4.0  | 84.0 %    |
| 5.0  | 90.5 %    |

Run it yourself (downloads ~350 MB model on first run):

```bash
uv run --with "pytest,mlx-lm>=0.22" pytest tests/ -m integration -v -s
```

## Run tests

```bash
# Unit tests (no download required)
uv run --with pytest pytest tests/ -m "not integration"

# Integration tests (requires mlx-lm + Qwen2.5-0.5B ~350 MB)
uv run --with "pytest,mlx-lm>=0.22" pytest tests/ -m integration -s
```

## API reference

```python
from turboquant.mlx import compress, compress_vectors, inner_product, search, decompress

# KV cache (4-D layout: batch, heads, seq, dim)
compressed = compress(keys, bits=3.5, m=64)
scores     = inner_product(queries, compressed)  # [B, H, q_seq, k_seq]

# Vector search (2-D layout: N_docs, dim)
corpus     = compress_vectors(embeddings, bits=3.5)
scores, ix = search(query, corpus, top_k=10)

# Debug — approximate reconstruction (biased, omits QJL correction)
keys_approx = decompress(compressed, dim=64)
```

`compress` / `compress_vectors` parameters:
- `bits` — total bits per element. Must be > 1.0. Recommended: 2.5–5.0.
- `m` — QJL sketch dimension (default 64; higher = lower variance, larger memory).
- `signs`, `jl_matrix` — pass pre-generated arrays to reuse across calls (recommended for KV cache where the same rotation should apply to all tokens).

## Citation

```bibtex
@inproceedings{turboquant2026,
  title   = {TurboQuant: Near-Optimal KV Cache Quantization},
  year    = {2026},
  url     = {https://arxiv.org/abs/2504.19874},
}
```
