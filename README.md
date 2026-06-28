# Forward-Only Local Adaptation of a Frozen INT4 Base — O(1)-in-Depth Memory

A reproducible benchmark of **gradient-free, backprop-free local adaptation** of a frozen
INT4-quantized network, measured head-to-head against LoRA. The adaptation runs entirely
forward — no backward pass, no autograd graph — so its per-step memory is **constant in
network depth**, where LoRA's grows linearly.

This is an **honest proof-of-concept on feedforward nets (digits, MNIST)**, with a documented
hard boundary: the method does not extend to attention (tested below). It is not a large-model
result. See *Where it breaks* and *Scope & limitations* before reading anything else into it.

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
| 4  | 0.185 (≈ chance) | 0.791 |
| 8  | 0.185 (≈ chance) | 0.741 |
| 16 | 0.189 (≈ chance) | 0.689 |

**It generalizes from digits (1.8k samples) to MNIST (60k)** — the ablation ordering holds:

![Accuracy by depth](assets/fig1_accuracy_by_depth.png)

| Depth L | no-adapt | LoRA | HALO+FOTON | +Precond |
|--------:|---------:|-----:|-----------:|---------:|
| 1  | 0.146 | 0.892 | 0.859 | 0.857 |
| 4  | 0.086 | 0.894 | 0.779 | 0.794 |
| 8  | 0.102 | 0.895 | 0.758 | 0.763 |
| 16 | 0.094 | 0.898 | 0.670 | 0.762 |

**The gap to LoRA is capacity-bound, not inherent** — it closes monotonically with adapter
rank, with diminishing returns, while the memory advantage *grows* (MNIST, L=8):

![Rank sweep](assets/fig2_rank_sweep.png)

| rank r | LoRA | HALO | gap | memory advantage |
|-------:|-----:|-----:|----:|-----------------:|
| 8  | 0.895 | 0.763 | 0.132 | 4.17× |
| 16 | 0.900 | 0.817 | 0.083 | 4.31× |
| 32 | 0.896 | 0.853 | 0.043 | 4.52× |
| 64 | 0.897 | 0.860 | 0.037 | 4.74× |

The memory advantage is measured as peak adapt-step bytes (LoRA retains one activation tensor
per layer for the backward graph — O(L); HALO holds one layer's worth at a time — O(1)). At
L=8 this is ~4–5×; it scales with depth, so a deeper stack widens the gap further.

## Where it breaks: attention (tested, not assumed)

The method works on feedforward layers. It does **not** carry to attention. This was tested
directly, not assumed — and the negative result is documented here on purpose, because knowing
where a method stops mattering is part of the method.

Task: single-head-solvable associative recall (key→value retrieval), the canonical job of
attention. Adapt only the Q/K/V projections of a frozen INT4 attention block:

| method | test accuracy |
|---|---|
| LoRA on Q/K/V (backprop) | **0.997** |
| no adaptation | 0.177 |
| HALO-DFA (no ortho) | 0.126 |
| HALO+FOTON | 0.147 |
| HALO+FOTON+Precond | 0.153 |
| random baseline | 0.125 |

All forward-only variants collapse to chance. Localization (adapting only the *linear* output
projection W_o, Q/K/V frozen) reaches just 0.199 — confirming the failure is not a tuning issue.

**Why:** DFA cannot assign credit through the softmax. In an MLP the input→output map of each
layer is linear before the nonlinearity, so a random-projected error times the local activation
is a usable update. In attention, Q and K act on the output *through the attention distribution*
— deeply nonlinear, across the whole sequence. Associative recall needs a precise query-key
match, which is exactly the signal backprop carries through the softmax gradient and DFA's random
projection does not. This is consistent with the known weakness of feedback-alignment methods on
attention; it is an open research problem, not a bug in this code.

`attn_halo_gate.py` reproduces this table; `attn_sanity.py` shows the same testbed reaches 100%
under full backprop (so the testbed is healthy — the gap is the method, not the task).

## Scope & limitations (read this)

- **Feedforward layers only.** Results are on fully-connected nets over digits and MNIST, and
  the attention test above shows the method does **not** extend to transformer Q/K/V. This is
  **not** a transformer or LLM result. Forward-only attention credit assignment is an open
  research direction, not something this repo claims to solve.
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

## Real-World BMS SOH Adaptation Benchmark (v2.0)

We extend HALO to a real-world energy-ML regression task: **on-device State-of-Health (SOH / cycle-life) adaptation** across two battery datasets under memory-constrained regimes:
1. **Severson LFP (Cycle-Life)**: Out-of-distribution (OOD) adaptation on the Severson Batch 3 cohort ($N_{\text{adapt}} \in [5, 10, 15, 20]$ cells).
2. **KIT NMC (SOH / Instance Split)**: Instance-split OOD adaptation on Instance 3 cells (Instance 1 & 2 Train) under Option C ($N_{\text{adapt}} \in [5, 10, 15, 20]$ cells).

We perform a statistically rigorous evaluation across **20 random seeds (0–19)** with strictly paired splits, base models, and cohorts. Optimal parameters are selected using deployable cross-validation (LOO-CV for $N_{\text{adapt}}=5$, 5-fold CV for $N_{\text{adapt}} > 5$) without test-set peeking.

### Headline Results (Rank 16, $N_{\text{adapt}} = 20$)

#### 1. Severson LFP Dataset (Cycle-Life)
| Method / Metric | MAPE (Mean ± Std) | MAPE (Median) | RMSE (cycles) | $R^2$ (Mean ± Std) | $R^2$ (Median) | Wilcoxon $p$-val | Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base (No Adapt)** | $20.14 \pm 3.15\%$ | $19.77\%$ | $301.8 \pm 59.7$ | $-0.50 \pm 0.53$ | $-0.39$ | — | — |
| **LoRA (Backprop)** | **$14.60 \pm 2.43\%$** | **$13.78\%$** | **$206.0 \pm 36.4$** | **$0.27 \pm 0.37$** | **$0.38$** | — | — |
| **HALO CV (Forward)** | $19.00 \pm 4.81\%$ | $18.65\%$ | $268.6 \pm 64.4$ | $-0.39 \pm 0.95$ | $-0.10$ | $0.0010$ | $[+2.45\%, +6.56\%]$ |
| **HALO Oracle (Best)** | $15.19 \pm 2.18\%$ | $15.24\%$ | $217.0 \pm 46.6$ | $0.16 \pm 0.50$ | $0.27$ | — | — |

#### 2. KIT NMC Dataset (SOH / Option C)
| Method / Metric | MAPE (Mean ± Std) | MAPE (Median) | RMSE (cycles) | $R^2$ (Mean ± Std) | $R^2$ (Median) | Wilcoxon $p$-val | Bootstrap 95% CI |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Base (No Adapt)** | $41.86 \pm 8.94\%$ | $39.91\%$ | $23.9 \pm 3.7$ | $0.55 \pm 0.10$ | $0.55$ | — | — |
| **LoRA (Backprop)** | **$36.17 \pm 8.50\%$** | **$35.30\%$** | **$23.2 \pm 6.2$** | **$0.54 \pm 0.26$** | **$0.63$** | — | — |
| **HALO CV (Forward)** | $59.90 \pm 25.31\%$ | $53.71\%$ | $40.4 \pm 37.9$ | $-1.87 \pm 8.79$ | $0.30$ | $< 10^{-5}$ | $[+14.42\%, +36.80\%]$ |
| **HALO Oracle (Best)** | $43.79 \pm 7.73\%$ | $42.24\%$ | $25.7 \pm 5.4$ | $0.46 \pm 0.23$ | $0.54$ | — | — |

*Bootstrap CI reports the 95% percentile interval on the paired difference `(HALO CV - LoRA)`.*

---

### Core Findings & Boundary Characterization

1. **The Adaptability Ceiling on OOD Shift (The $R^2$ Reality)**:
   In this benchmark, the **base model (No Adapt)** is a **frozen random INT4 network with a low-rank adapter pre-trained offline on Batch 1 & 2**. This is a highly constrained, memory-saving architecture designed for microcontroller BMS, not an offline optimal predictor.
   - On the **Severson Cohort OOD Shift (Batch 3)**, this rigid base model is structurally weak, yielding a negative base $R^2$ of $-0.50 \pm 0.53$ (Median $-0.39$), meaning its zero-adaptation predictions perform worse than predicting the evaluation mean. 
   - **LoRA** is the only method reaching a positive median $R^2$ ($0.38$), though transfer remains weak for all methods under this OOD cohort shift; no method achieves reliable transfer. Forward-only **HALO CV** remains negative on average at $-0.39 \pm 0.95$ (Median $-0.10$). Thus, for OOD cohort shift, HALO CV fails to escape the negative $R^2$ regime.
   - On the **KIT NMC Instance Split (Option C)**, the domain shift is milder (same testing conditions, different cell instances). Hence, the base model transfers with a positive $R^2$ of $0.55$. LoRA improves the median $R^2$ to $0.63$, whereas HALO CV experiences a catastrophic collapse on specific seeds ($R^2 = -1.87 \pm 8.79$, though the median is positive at $0.30$).

2. **Sanity Check vs. Adaptation Benchmark**:
   - The offline feature sanity check (which reports $R^2 \approx 0.70\text{--}0.90$ across datasets) was evaluated using a **fully-trained, unconstrained model** on the extracted features. This proved that the features contain a strong degradation signal.
   - The adaptation benchmark evaluates the **hardware-constrained random-base adapter setup**. The lower/negative $R^2$ scores indicate the structural cost of frozen random INT4 projection under OOD domain shifts, not a failure of the feature pipeline itself.

3. **Catastrophic Collapse of Forward-Only DFA (Medians vs. Means)**:
   The huge variance for HALO CV on KIT NMC ($R^2$ std of $\pm 8.79$, MAPE std of $\pm 25.31\%$) is driven by severe failures on specific seeds, where the gradient-free Direct Feedback Alignment update rule explodes. The positive median ($0.30$) masks a disqualifying tail: forward-only DFA collapses catastrophically on a subset of seeds ($R^2$ down to large-negative, std $\pm 8.79$). For on-device deployment this tail risk is the finding — median stability is not sufficient when worst-case behavior is unbounded.

4. **Honest Metric Integrity (KIT NMC MAPE Inflation)**:
   The absolute MAPE is elevated on KIT NMC for all models (including Base and LoRA). This is not a feature pipeline or model failure, but a mathematical consequence of small cycle-life denominators (NMC cells degrade fast, with lifetimes of $10\text{--}20$ cycles, where a 3-cycle error represents a $20\text{--}30\%$ MAPE). This is verified by:
   - **Zero Correlation**: The cell lifetime $\leftrightarrow$ absolute error correlation is $-0.0290$ (practically zero).
   - **Temperature Group Analysis**: $0^\circ\text{C}$ cells, which have the shortest lifetimes, show the *lowest* average MAPE ($18.55\%$), contradicting any pipeline bug.
   - **Stable Signal**: Base model $R^2$ on OOD cells is $0.70$ (train $R^2 = 0.75$), confirming the features contain a strong predictive signal. RMSE and $R^2$ report the true physical picture.

All SOH adaptation code is fully reproducible via `rigorous_halo_bess_adaptation.py` (Severson) and `run_kit_benchmark.py` (KIT). Result JSON files are persisted in `verified_results.json` and `kit_verified_results.json`.

## Reproduce

```bash
pip install torch numpy scikit-learn matplotlib scipy
python reproduce.py                     # MNIST & digits classification
python rigorous_halo_bess_adaptation.py # BESS SOH adaptation
```

`reproduce.py` downloads MNIST from a GitHub mirror and runs all three experiments (FOTON necessity, MNIST depth ablation, rank capacity) from a single seed with no shared state.

## References

- Nøkland, *Direct Feedback Alignment Provides Learning in Deep Neural Networks*, NeurIPS 2016
- Fagnou et al., *Forward Only Learning for Orthogonal Neural Networks of Any Depth*, arXiv:2512.20668 (2025)
- Jordan, *Muon optimizer* (Newton–Schulz orthogonalization), 2024
- Gupta et al., *Deep Learning with Limited Numerical Precision* (stochastic rounding), ICML 2015
- Oja, *Simplified neuron model as a principal component analyzer*, 1982

Related work using Oja's rule for a *different* problem (KV-cache compression, not weight
adaptation): Zhu et al., *OjaKV*, arXiv:2509.21623 (2025) — cited for completeness, not part of
this method.
