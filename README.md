# CIMA: Continuous Implicit Manifold Attention

A lightweight, high-accuracy KV cache compression architecture for long-context LLMs.

---

About a year ago, I was feeding a 500-page book into ChatGPT to answer specific questions. Like anyone working with long contexts knows, LLMs start dropping details or running out of memory because the Key-Value (KV) cache grows massive—storing raw data for every single token eats up GPU VRAM instantly.

Standard fixes usually try to solve this by either:
1. **Throwing away tokens** (which drops critical details).
2. **Heavy quantization** (which turns memory into low-precision mush).

I thought "What if instead of saving millions of individual data points, we map the sequence along a timeline and draw a smooth mathematical curve through the topics?"


## How It Works

Think of standard attention like writing millions of individual sticky notes for every word in a book. Eventually, you wont have space.

CIMA replaces those sticky notes with two things:
1. **A Continuous Curve:** A lightweight Sinusoidal Representation Network (SIREN) maps the entire sequence continuous-style. Storing the formula for one curve takes almost zero space compared to gigabytes of raw tokens.
2. **2% Precision Anchors:** Because a smooth curve might blur past tiny high-entropy details (like a specific key password or rare fact on page 342), CIMA pinpoints the top 2% highest-error tokens and keeps exact copies of them.

By combining smooth math with pinpoint anchors, you get massive memory savings without losing a single "needle in a haystack."

Key Results

Tested on sequence lengths up to **32,000 tokens**:

- **97.4% VRAM Footprint Reduction:** Cut 32k context KV cache memory from 3.5 GB down to **91.38 MB**.
- **100% Retrieval Accuracy:** Zero precision loss on Needle-In-A-Haystack (NIAH) benchmarks.
- **Drop-in Compatible:** Subclasses Hugging Face `DynamicCache` for native generation pipelines (validated on `Qwen2.5-0.5B`).

---

## Citation & Paper

Published as an open-access preprint on Zenodo:

**Title:** Continuous Implicit Manifold Attention (CIMA): Sub-Linear KV Cache Memory Scaling via Neural Fields and Residual Anchors  
**DOI:** [10.5281/zenodo.23049831](https://zenodo.org/records/23049831)

```bibtex
@article{sharvesh2026cima,
  title={Continuous Implicit Manifold Attention (CIMA): Sub-Linear KV Cache Memory Scaling via Neural Fields and Residual Anchors},
  author={Sharvesh},
  journal={Zenodo Preprint},
  doi={10.5281/zenodo.23049831},
  year={2026}
}
