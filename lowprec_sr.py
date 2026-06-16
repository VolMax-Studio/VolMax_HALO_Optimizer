"""
VolMax HALO — Full Low-Precision Update Test (Stochastic Rounding)
====================================================================
Closes the gap: until now base weights were INT4 but the HALO adapters
(U, V) lived in FP32. A real edge-training claim requires the UPDATE
itself to survive low precision. We quantize the adapter accumulation
to a coarse grid and compare:

  RTN : round-to-nearest      -- expected to collapse (underflow to 0)
  SR  : stochastic rounding   -- expected to preserve convergence

Stochastic rounding (Gupta et al. 2015):
  SR(x) = floor(x)   w.p. 1 - frac(x)
        = ceil(x)    w.p. frac(x)
unbiased: E[SR(x)] = x, so tiny sub-LSB deltas survive probabilistically
over many steps via the law of large numbers, without a high-precision
accumulator.

Gate: under a coarse adapter grid, does SR recover accuracy that RTN
loses? If yes, the low-precision edge story holds. If no, report plainly.
"""

import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
np.random.seed(42)

dig = load_digits()
Xall = StandardScaler().fit_transform(dig.data).astype(np.float32)
yall = dig.target.astype(np.int64)
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=0, stratify=yall)
Xtr = torch.tensor(Xtr); ytr = torch.tensor(ytr)
Xte = torch.tensor(Xte); yte = torch.tensor(yte)

D = 64; N_CLASS = 10; BATCH = 64; STEPS = 600
DEPTHS = [1, 2, 4, 8, 16]; N_BITS = 4; RANK = 8

# Adapter quantization grid: coarse enough to trigger underflow with RTN.
# INT8-ish dynamic grid for the adapter weights themselves.
ADAPTER_LEVELS = 15           # ~4-bit signed grid for adapter weights
ADAPTER_RANGE = 1.0           # adapters clamped to [-1, 1]
ADAPTER_STEP = (2 * ADAPTER_RANGE) / ADAPTER_LEVELS  # quantization step

class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n_bits):
        qmin, qmax = -(2 ** (n_bits - 1)), (2 ** (n_bits - 1) - 1)
        scale = x.abs().max().clamp(min=1e-8) / qmax
        return torch.clamp(torch.round(x / scale), qmin, qmax) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None

def fq(x, n_bits=N_BITS):
    return FakeQuantSTE.apply(x, n_bits)

def build_frozen_base(depth, d):
    torch.manual_seed(1)
    return [(fq(torch.randn(d, d) * (1.0 / np.sqrt(d))), fq(torch.zeros(d))) for _ in range(depth)]

def quantize_rtn(w):
    """Round-to-nearest onto the adapter grid."""
    q = torch.round(w / ADAPTER_STEP)
    return torch.clamp(q * ADAPTER_STEP, -ADAPTER_RANGE, ADAPTER_RANGE)

def quantize_sr(w):
    """Stochastic rounding onto the adapter grid (unbiased)."""
    scaled = w / ADAPTER_STEP
    floor = torch.floor(scaled)
    frac = scaled - floor
    rnd = (torch.rand_like(frac) < frac).float()   # ceil with prob frac
    q = floor + rnd
    return torch.clamp(q * ADAPTER_STEP, -ADAPTER_RANGE, ADAPTER_RANGE)

def ortho_ns_muon(W, iters=5):
    if not torch.isfinite(W).all():
        W = torch.randn(W.shape[0], W.shape[1]) * 0.05
    a, b, c = 3.4445, -4.7750, 2.0315
    X = W / (W.norm() + 1e-8)
    for _ in range(iters):
        A = X.t() @ X
        X = a * X + b * (X @ A) + c * (X @ (A @ A))
    return X

def softmax(z):
    z = z - z.max(1, keepdim=True).values
    e = z.exp(); return e / e.sum(1, keepdim=True)

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def run_halo_lowprec(depth, d, quant_fn, lr=0.03, use_ortho=True):
    layers = build_frozen_base(depth, d)
    torch.manual_seed(2)
    U = [quant_fn(torch.randn(d, RANK) * 0.05) for _ in range(depth)]
    V = [quant_fn(torch.randn(RANK, d) * 0.05) for _ in range(depth)]
    Wout = torch.randn(d, N_CLASS) * (1.0 / np.sqrt(d))   # readout stays FP32 (small)
    torch.manual_seed(7)
    R = [torch.randn(N_CLASS, d) * (1.0 / np.sqrt(N_CLASS)) for _ in range(depth)]

    def fwd_full(X):
        h = X; acts = []
        for i, (W, b) in enumerate(layers):
            a_in = h
            a_out = torch.tanh(a_in @ W + (a_in @ U[i]) @ V[i] + b)
            acts.append((a_in, a_out)); h = a_out
        return h @ Wout, acts

    clip = 0.5; wout_lr = 0.03
    for step in range(STEPS):
        idx = torch.randint(0, Xtr.shape[0], (BATCH,))
        xb, yb = Xtr[idx], ytr[idx]
        with torch.no_grad():
            logits, acts = fwd_full(xb)
            p = softmax(logits)
            onehot = torch.zeros_like(p); onehot[torch.arange(BATCH), yb] = 1.0
            global_err = (p - onehot)
            a_L = acts[-1][1]
            Wout -= wout_lr * ((a_L.t() @ global_err) / BATCH).clamp(-clip, clip)
            for i, (W, b) in enumerate(layers):
                a_in, a_out = acts[i]
                e_local = global_err @ R[i]
                delta = e_local * (1.0 - a_out ** 2)
                z = a_in @ U[i]
                dU = ((a_in.t() @ z) / BATCH - (a_in ** 2).mean(0).unsqueeze(1) * U[i]).clamp(-clip, clip)
                dV = ((z.t() @ delta) / BATCH).clamp(-clip, clip)
                # Apply update in continuous space, then re-quantize the
                # adapter weight onto the coarse grid with the chosen rule.
                U[i] = quant_fn(U[i] + lr * dU)
                V[i] = quant_fn(V[i] - lr * dV)
                if use_ortho:
                    # ortho in continuous space, then re-quantize
                    U[i] = quant_fn(ortho_ns_muon(U[i]))

    with torch.no_grad():
        logits, _ = fwd_full(Xte)
        return accuracy(logits, yte)

print("=" * 92)
print("VolMax HALO — Full INT4-adapter update: Round-to-Nearest vs Stochastic Rounding")
print(f"Adapter grid: {ADAPTER_LEVELS} levels, step={ADAPTER_STEP:.4f} over [-{ADAPTER_RANGE},{ADAPTER_RANGE}]")
print("=" * 92)
print(f"{'L':>3} | {'FP32 (ref)':>10} | {'INT4 RTN':>10} | {'INT4 SR':>10}")

# FP32 reference (no adapter quantization) for comparison
def run_fp32(depth, d, lr=0.03):
    return run_halo_lowprec(depth, d, quant_fn=lambda x: x, lr=lr)

for L in DEPTHS:
    acc_fp32 = run_fp32(L, D)
    acc_rtn = run_halo_lowprec(L, D, quantize_rtn)
    acc_sr = run_halo_lowprec(L, D, quantize_sr)
    print(f"{L:3d} | {acc_fp32:10.3f} | {acc_rtn:10.3f} | {acc_sr:10.3f}")

# Direct underflow demonstration: how many adapter updates die under RTN?
print("\nUnderflow demonstration (fraction of adapter deltas that round to 0):")
torch.manual_seed(0)
deltas = torch.randn(10000) * 0.002   # typical small Oja deltas
rtn_dead = (quantize_rtn(deltas) == 0).float().mean().item()
sr_survived = 1.0 - (quantize_sr(deltas) == 0).float().mean().item()
print(f"  RTN: {rtn_dead*100:.1f}% of deltas collapse to exactly 0 (dead update)")
print(f"  SR:  {sr_survived*100:.1f}% of deltas survive as a nonzero quantum (probabilistic)")
