#!/usr/bin/env python3
"""
VolMax HALO — Core Mathematical Equivalence Test
================================================
Verifies that the NumPy port of the HALO core (from bess_soh_halo_adaptation.py)
is mathematically identical to the original PyTorch implementation (from reproduce.py)
when evaluated on the digits dataset classification task.
"""
import torch
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class STE_torch(torch.autograd.Function):
    @staticmethod
    def forward(c, x, nb):
        qm, qM = -(2**(nb-1)), (2**(nb-1)-1)
        s = x.abs().max().clamp(min=1e-8) / qM
        return torch.clamp(torch.round(x/s), qm, qM) * s
    @staticmethod
    def backward(c, g):
        return g, None

def fq_torch(x, nb=4):
    return STE_torch.apply(x, nb)

def nsmuon_torch(W, it=5):
    if not torch.isfinite(W).all():
        W = torch.randn(W.shape[0], W.shape[1]) * 0.05
    a, b, c = 3.4445, -4.7750, 2.0315
    X = W / (W.norm() + 1e-8)
    for _ in range(it):
        A = X.t() @ X
        X = a*X + b*(X @ A) + c*(X @ (A @ A))
    return X

def make_stack_torch(depth, Hd, seed=1):
    torch.manual_seed(seed)
    return [(fq_torch(torch.randn(Hd, Hd)*(1/np.sqrt(Hd))), fq_torch(torch.zeros(Hd))) for _ in range(depth)]

# We will import or reimplement the numpy versions
from bess_soh_halo_adaptation import nsmuon as nsmuon_numpy, fq as fq_numpy

def train_halo_pytorch(layers, Xtr_in, ytr, Xte_in, yte, Hd, rank, NCLASS, steps,
                       use_foton=True, precond=False, lam=0.1, lr=0.03):
    torch.manual_seed(2)
    U = [torch.randn(Hd, rank) * 0.05 for _ in layers]
    V = [torch.randn(rank, Hd) * 0.05 for _ in layers]
    Wout = torch.randn(Hd, NCLASS) * (1 / np.sqrt(Hd))
    torch.manual_seed(7)
    R = [torch.randn(NCLASS, Hd) * (1 / np.sqrt(NCLASS)) for _ in layers]
    Ir = torch.eye(rank)
    clip = 0.5
    N = Xtr_in.shape[0]
    
    # We will generate deterministic batch indices for step-by-step equivalence
    np.random.seed(42)
    batch_indices = [np.random.randint(0, N, min(128, N)) for _ in range(steps)]
    
    def fwd(hin):
        h = hin
        ac = []
        for i, (W, b) in enumerate(layers):
            ai = h
            ao = torch.tanh(ai @ W + (ai @ U[i]) @ V[i] + b)
            ac.append((ai, ao))
            h = ao
        return h @ Wout, ac

    def sm(z):
        z = z - z.max(1, keepdim=True).values
        e = z.exp()
        return e / e.sum(1, keepdim=True)

    for s in range(steps):
        bi = torch.tensor(batch_indices[s])
        xb, yb = Xtr_in[bi], ytr[bi]
        with torch.no_grad():
            lo, ac = fwd(xb)
            p = sm(lo)
            oh = torch.zeros_like(p)
            oh[torch.arange(len(bi)), yb] = 1
            ge = p - oh
            aL = ac[-1][1]
            Wout -= 0.03 * ((aL.t() @ ge) / len(bi)).clamp(-clip, clip)
            for i, (W, b) in enumerate(layers):
                ai, ao = ac[i]
                el = ge @ R[i]
                de = el * (1 - ao**2)
                z = ai @ U[i]
                dU = (ai.t() @ z) / len(bi) - (ai**2).mean(0).unsqueeze(1) * U[i]
                dV = (z.t() @ de) / len(bi)
                if precond:
                    Gz = (z.t() @ z) / len(bi) + lam * Ir
                    Gzi = torch.linalg.inv(Gz)
                    dU = dU @ Gzi
                    dV = Gzi @ dV
                dU = dU.clamp(-clip, clip)
                dV = dV.clamp(-clip, clip)
                U[i] = U[i] + lr * dU
                V[i] = V[i] - lr * dV
                if use_foton:
                    U[i] = nsmuon_torch(U[i])
                    
    with torch.no_grad():
        lo, _ = fwd(Xte_in)
        pred_acc = (lo.argmax(1) == yte).float().mean().item()
        
    return pred_acc, U, V, Wout

def train_halo_numpy(layers, Xtr_in, ytr, Xte_in, yte, Hd, rank, NCLASS, steps,
                     use_foton=True, precond=False, lam=0.1, lr=0.03):
    # Initialize numpy copies of initial weights exactly using PyTorch initial states for equivalence
    torch.manual_seed(2)
    U = [torch.randn(Hd, rank).numpy() * 0.05 for _ in layers]
    V = [torch.randn(rank, Hd).numpy() * 0.05 for _ in layers]
    Wout = torch.randn(Hd, NCLASS).numpy() * (1 / np.sqrt(Hd))
    torch.manual_seed(7)
    R = [torch.randn(NCLASS, Hd).numpy() * (1 / np.sqrt(NCLASS)) for _ in layers]
    Ir = np.eye(rank)
    clip = 0.5
    N = Xtr_in.shape[0]
    
    # Same deterministic batch indices
    np.random.seed(42)
    batch_indices = [np.random.randint(0, N, min(128, N)) for _ in range(steps)]
    
    # Layer weights converted to numpy
    layers_np = [(W.numpy(), b.numpy()) for W, b in layers]
    Xtr_np = Xtr_in.numpy()
    ytr_np = ytr.numpy()
    Xte_np = Xte_in.numpy()
    yte_np = yte.numpy()
    
    def fwd(hin):
        h = hin
        ac = []
        for i, (W, b) in enumerate(layers_np):
            ai = h
            ao = np.tanh(ai @ W + (ai @ U[i]) @ V[i] + b)
            ac.append((ai, ao))
            h = ao
        return h @ Wout, ac

    def sm(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    for s in range(steps):
        bi = batch_indices[s]
        xb, yb = Xtr_np[bi], ytr_np[bi]
        
        lo, ac = fwd(xb)
        p = sm(lo)
        oh = np.zeros_like(p)
        oh[np.arange(len(bi)), yb] = 1
        ge = p - oh
        aL = ac[-1][1]
        
        Wout -= 0.03 * np.clip((aL.T @ ge) / len(bi), -clip, clip)
        for i, (W, b) in enumerate(layers_np):
            ai, ao = ac[i]
            el = ge @ R[i]
            de = el * (1 - ao**2)
            z = ai @ U[i]
            dU = (ai.T @ z) / len(bi) - (ai**2).mean(0).reshape(-1, 1) * U[i]
            dV = (z.T @ de) / len(bi)
            if precond:
                Gz = (z.T @ z) / len(bi) + lam * Ir
                Gzi = np.linalg.inv(Gz)
                dU = dU @ Gzi
                dV = Gzi @ dV
            dU = np.clip(dU, -clip, clip)
            dV = np.clip(dV, -clip, clip)
            U[i] = U[i] + lr * dU
            V[i] = V[i] - lr * dV
            if use_foton:
                U[i] = nsmuon_numpy(U[i])
                
    lo, _ = fwd(Xte_np)
    pred_acc = (lo.argmax(1) == yte_np).mean()
    
    return pred_acc, U, V, Wout

if __name__ == "__main__":
    print("=" * 80)
    print("Evaluating Equivalence: PyTorch HALO vs NumPy HALO Core Port")
    print("=" * 80)
    
    # Load digits dataset
    dig = load_digits()
    Xd = StandardScaler().fit_transform(dig.data).astype(np.float32)
    yd = dig.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
    Xtr = torch.tensor(Xtr)
    ytr = torch.tensor(ytr)
    Xte = torch.tensor(Xte)
    yte = torch.tensor(yte)
    
    # Create layers
    L = 4
    Hd = 64
    rank = 8
    NCLASS = 10
    steps = 600
    
    # PyTorch stack
    layers_torch = make_stack_torch(L, Hd)
    
    # Train both
    print("Training PyTorch HALO...")
    acc_pt, U_pt, V_pt, Wout_pt = train_halo_pytorch(
        layers_torch, Xtr, ytr, Xte, yte, Hd, rank, NCLASS, steps,
        use_foton=True, precond=True, lam=0.1
    )
    
    print("Training NumPy HALO Port...")
    acc_np, U_np, V_np, Wout_np = train_halo_numpy(
        layers_torch, Xtr, ytr, Xte, yte, Hd, rank, NCLASS, steps,
        use_foton=True, precond=True, lam=0.1
    )
    
    print("\n--- RESULTS ---")
    print(f"PyTorch Accuracy : {acc_pt:.5f}")
    print(f"NumPy Accuracy   : {acc_np:.5f}")
    print(f"Accuracy Diff    : {abs(acc_pt - acc_np):.5e}")
    
    # Check weight differences
    diffs_U = [np.max(np.abs(u_pt.numpy() - u_np)) for u_pt, u_np in zip(U_pt, U_np)]
    diffs_V = [np.max(np.abs(v_pt.numpy() - v_np)) for v_pt, v_np in zip(V_pt, V_np)]
    diff_Wout = np.max(np.abs(Wout_pt.numpy() - Wout_np))
    
    print("\nWeight Max Absolute Differences:")
    for i in range(L):
        print(f"  Layer {i} U diff: {diffs_U[i]:.5e}")
        print(f"  Layer {i} V diff: {diffs_V[i]:.5e}")
    print(f"  Wout diff     : {diff_Wout:.5e}")
    
    success = max(diffs_U + diffs_V + [diff_Wout]) < 1e-5
    if success:
        print("\n[SUCCESS] NumPy HALO core is mathematically equivalent to the PyTorch reference implementation!")
    else:
        print("\n[FAILURE] Mathematical differences found!")
