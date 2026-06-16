# Forward-Only Local Adaptation of a Frozen INT4 Base — O(1)-in-Depth Memory

A reproducible benchmark of **gradient-free, backprop-free local adaptation** of a frozen
INT4-quantized network, measured head-to-head against LoRA. The adaptation runs entirely
forward — no backward pass, no autograd graph — so its per-step memory is **constant in
network depth**, where LoRA's grows linearly.

This is an **honest proof-of-concept on MLPs (digits, MNIST)**, not a large-model result.
See *Scope & limitations* before reading anything else into it.

---

## What this is

A small adaptation layer ("HALO") sits on top of a frozen INT4 base and is updated by a
**three-part forward-only rule**, each part a known technique, combined and ablated here:

1. **DFA** (Direct Feedback Alignment, Nøkland 2016) — the global task error is projected
   onto each layer by a fixed random matrix, delivering a task signal to every layer
   without backpropagation.
2. **FOTON orthogonalization** (Fagnou et al. 2025, arXiv:2512.20668) — the adapter's input
   factor is kept orthogonal (via Newton–Schulz / Björck), which is what keeps the
   forward-only signal from collapsing as depth grows.
3. **Newton–Muon right preconditioning** — the update is whitened by the inverse
   second-moment of the bottleneck activations `(zᵀz + λI)⁻¹`, which closes part of the
   remaining accuracy gap to LoRA.

The contribution here is **the integration, the benchmark, and the ablations** — not the
individual techniques.

## Headline results (all regenerated from scratch by `reproduce.py`)

**Each component is necessary** — remove FOTON orthogonalization and the forward-only signal
collapses to chance in depth (digits):

| Depth L | plain DFA (no ortho) | with FOTON |
|--------:|---------------------:|-----------:|
| 4  | 0.191 (≈ chance) | 0.791 |
| 8  | 0.193 (≈ chance) | 0.741 |
| 16 | 0.193 (≈ chance) | 0.693 |

**It generalizes from digits (1.8k samples) to MNIST (60k)** — the ablation ordering holds:

![Accuracy by depth](assets/fig1_accuracy_by_depth.png)

| Depth L | no-adapt | LoRA | HALO+FOTON | +Precond |
|--------:|---------:|-----:|-----------:|---------:|
| 1  | 0.146 | 0.892 | 0.859 | 0.857 |
| 4  | 0.086 | 0.894 | 0.778 | 0.794 |
| 8  | 0.102 | 0.895 | 0.751 | 0.763 |
| 16 | 0.094 | 0.898 | 0.681 | 0.762 |

**The gap to LoRA is capacity-bound, not inherent** — it closes monotonically with adapter
rank, with diminishing returns, while the memory advantage *grows* (MNIST, L=8):

![Rank sweep](assets/fig2_rank_sweep.png)

| rank r | LoRA | HALO | gap | memory advantage |
|-------:|-----:|-----:|----:|-----------------:|
| 8  | 0.895 | 0.763 | 0.132 | 4.17× |
| 16 | 0.900 | 0.817 | 0.083 | 4.31× |
| 32 | 0.896 | 0.856 | 0.040 | 4.52× |
| 64 | 0.897 | 0.859 | 0.038 | 4.74× |

The memory advantage is measured as peak adapt-step bytes (LoRA retains one activation tensor
per layer for the backward graph — O(L); HALO holds one layer's worth at a time — O(1)). At
L=8 this is ~4–5×; it scales with depth, so a deeper stack widens the gap further.

## Scope & limitations (read this)

- **MLPs only.** Results are on fully-connected nets over digits and MNIST. This is **not** a
  transformer or LLM result. The next honest step toward any large-model claim is adapting a
  single real transformer layer.
- **MNIST at ~0.86–0.90 is MLP-ceiling, not SOTA** — CNNs reach 0.99. LoRA's ~0.89 here is the
  fair backprop ceiling *for this architecture*, the right comparison point, not an absolute.
- **LoRA still wins on accuracy** at equal rank. The pitch is memory, not accuracy: forward-only
  adaptation that holds O(1) memory in depth, at an accuracy cost that shrinks with rank.
- **λ for preconditioning is tuned per depth** (values in `reproduce.py`). They transfer from
  digits to MNIST without re-tuning, but they are not parameter-free.
- **Low-precision update caveat:** stochastic rounding of the *adapter itself* only helps on a
  fine grid (~INT8); on a coarse INT4 grid it injects more noise than signal. Adapters here are
  kept in higher precision; the *base* is INT4. (See commit history / `lowprec_sr.py`.)
- Trained on a 10k MNIST subset for runtime; full 60k would shift numbers marginally.

## Reproduce

```bash
pip install torch numpy scikit-learn matplotlib
python reproduce.py        # regenerates every number above in one run
```

`reproduce.py` downloads MNIST from a GitHub mirror and runs all three experiments
(FOTON necessity, MNIST depth ablation, rank capacity) from a single seed with no shared state.

## References

- Nøkland, *Direct Feedback Alignment Provides Learning in Deep Neural Networks*, NeurIPS 2016
- Fagnou et al., *Forward Only Learning for Orthogonal Neural Networks of Any Depth*, arXiv:2512.20668 (2025)
- Jordan, *Muon optimizer* (Newton–Schulz orthogonalization), 2024
- Gupta et al., *Deep Learning with Limited Numerical Precision* (stochastic rounding), ICML 2015
- Oja, *Simplified neuron model as a principal component analyzer*, 1982

Related work using Oja's rule for a *different* problem (KV-cache compression, not weight
adaptation): Zhu et al., *OjaKV*, arXiv:2509.21623 (2025) — cited for completeness, not part of
this method.
