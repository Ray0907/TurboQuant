# TurboQuant

Extreme KV cache compression for large language models — **10× smaller, zero accuracy loss, no fine-tuning**.

Implementation of three theoretically-grounded quantization algorithms from Google Research (ICLR 2026):

- **Random Rotation** — redistributes energy across dimensions for uniform quantization
- **PolarQuant** — converts vectors to polar coordinates, quantizing angles instead of Cartesian values (eliminates per-block memory overhead)
- **QJL** — 1-bit Johnson-Lindenstrauss residual correction for unbiased attention scores

## How It Works

### Step 1 — Random Rotation

Multiplying a KV vector by a random rotation matrix **R** spreads its energy uniformly across all dimensions, making each component equally quantizable.

![Random Rotation](assets/rotation.gif)

### Step 2 — PolarQuant

Pairs of coordinates are converted from Cartesian (x, y) to polar (r, θ). Because post-rotation angles are near-uniformly distributed, a fixed angular grid requires **zero per-block normalization overhead** — unlike traditional scalar quantization.

![PolarQuant](assets/polarquant.gif)

### Full Pipeline

Random Rotation → PolarQuant → QJL (1-bit residual): 32-bit KV vectors compressed to ~3 bits.

![TurboQuant Pipeline](assets/pipeline.gif)

## Results

| Method | Bits | LongBench Score | KV Memory |
|--------|------|----------------|-----------|
| FP16 baseline | 16 | 100% | 1× |
| KIVI | 4 | ~98% | 4× |
| PolarQuant | 4 | ~99% | **4×** |
| **TurboQuant** | **3** | **~100%** | **6×** |

- 4-bit TurboQuant achieves up to **8× speedup** in attention logit computation on H100
- Evaluated on LongBench, Needle-in-a-Haystack, RULER, ZeroSCROLLS, L-Eval
- Tested on Llama-3.1-8B, Gemma, Mistral — **no fine-tuning required**

## References

- [TurboQuant (ICLR 2026)](https://arxiv.org/pdf/2504.19874)
- [QJL: 1-Bit Quantized JL Transform (NeurIPS 2024)](https://arxiv.org/pdf/2406.03482)
- [PolarQuant (AISTATS 2026)](https://arxiv.org/pdf/2502.02617)
- [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression)
