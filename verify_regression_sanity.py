#!/usr/bin/env python3
"""
VolMax HALO — Regression Adaptation Sanity Test
===============================================
Verifies that the MSE regression adaptation signal 'ge = pred - y'
correctly drives the HALO optimizer to convergence on a synthetic regression problem.
"""
import numpy as np
from bess_soh_halo_adaptation import (make_stack, embed, halo_adapt, predict, fq)

def run_regression_sanity():
    print("=" * 80)
    print("Regression Sanity: Verifying HALO MSE Update Rule on Synthetic Task")
    print("=" * 80)
    
    np.random.seed(42)
    N = 500
    d = 10
    H = 64
    r = 8
    L = 2
    
    # 1. Generate synthetic features and targets
    X = np.random.randn(N, d)
    Win = np.random.randn(d, H) * (1 / np.sqrt(d))
    bin_ = np.zeros(H)
    
    # Embed features in activation space
    e = np.tanh(X @ Win + bin_)
    
    # Create target y = e @ w_star (exact solution exists)
    w_star = np.random.randn(H, 1)
    y = e @ w_star
    
    # 2. Setup frozen quantized base network layers
    layers = make_stack(L, H, seed=123)
    
    # Initialize adapters
    U = [np.random.randn(H, r) * 0.01 for _ in range(L)]
    V = [np.random.randn(r, H) * 0.01 for _ in range(L)]
    # Intentionally bad initial Wout to test adaptation
    Wout = np.random.randn(H, 1) * 0.1
    
    # Initial loss
    pred_init = predict(e, layers, U, V, Wout)
    loss_init = np.mean((pred_init - y) ** 2)
    print(f"Initial MSE Loss: {loss_init:.6f}")
    
    # 3. Adapt using HALO
    steps = 3000
    # Use larger learning rate to ensure fast convergence for sanity check
    U_opt, V_opt, Wout_opt, _ = halo_adapt(
        e, y, layers, U, V, Wout, H, r, steps,
        lr=0.08, batch=128, use_foton=True, precond=True, lam=0.1
    )
    
    # Final loss
    pred_final = predict(e, layers, U_opt, V_opt, Wout_opt)
    loss_final = np.mean((pred_final - y) ** 2)
    print(f"Final MSE Loss  : {loss_final:.6f}")
    
    ratio = loss_final / loss_init
    print(f"Loss Ratio      : {ratio:.6f} ({100*(1-ratio):.2f}% reduction)")
    
    # If loss drops by more than 80% and ends up at a low value, it works.
    success = loss_final < 0.5 and ratio < 0.2
    if success:
        print("\n[SUCCESS] HALO regression update rule successfully drives convergence!")
    else:
        print("\n[FAILURE] Regression adaptation failed to converge.")

if __name__ == "__main__":
    run_regression_sanity()
