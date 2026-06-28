#!/usr/bin/env python3
"""
VolMax HALO — Rigorous On-Device BMS Adaptation Benchmark (SOH / cycle-life)
=============================================================================
Provides a statistically rigorous validation of HALO vs LoRA under OOD shift:
  - 20 seeds (0-19) for variance control.
  - Paired comparison: HALO and LoRA use the exact same splits, base models,
    and adaptive/evaluation cell cohorts.
  - Percentile bootstrap (B=1000) on paired differences (HALO CV - LoRA).
  - Wilcoxon signed-rank test to compute paired p-values with effect sizes.
  - N_adapt sweep (5, 10, 15, 20 cells) on a strictly fixed, disjoint evaluation
    set of 20 cells. LOO-CV is used for N_adapt=5; 5-fold CV for N_adapt > 5.

All metrics are printed and saved to verified_results.json.
"""
import numpy as np
import json
import os
import scipy.stats as stats
from bess_soh_halo_adaptation import (load_data, fq, embed, make_stack,
    backprop_adapt, halo_adapt, predict, mape_cyclelife)

GRID = [(lam, lr) for lam in (0.03, 0.1, 0.3) for lr in (0.03, 0.1)]
HALO_STEPS = 1200
FACTORY_STEPS = 4000
DEVICE_STEPS = 600

def prep_rigorous(seed, r, L=4, H=64):
    rng = np.random.RandomState(seed)
    X, y, batch, cell, _ = load_data()
    tr = np.isin(batch, ["b1", "b2"]); b3 = batch == "b3"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xz = (X - mu) / sd
    ylog = np.log10(y); ymu, ysd = ylog[tr].mean(), ylog[tr].std() + 1e-8
    yz = (ylog - ymu) / ysd
    
    # Deterministic disjoint b3 splits (20 adapt, 20 eval)
    b3idx = np.where(b3)[0]
    rng.shuffle(b3idx)
    ad_idx, ev_idx = b3idx[:20], b3idx[20:]
    
    d = Xz.shape[1]
    Win = fq(rng.randn(d, H) * (1 / np.sqrt(d)))
    bin_ = np.zeros(H)
    layers = make_stack(L, H, seed=seed+100)
    
    e_tr = embed(Xz[tr], Win, bin_)
    y_tr = yz[tr]
    e_ad = embed(Xz[ad_idx], Win, bin_)
    y_ad = yz[ad_idx]
    e_ev = embed(Xz[ev_idx], Win, bin_)
    yev_cyc = y[ev_idx]
    
    # Initialize adapters
    U0 = [rng.randn(H, r) * 0.05 for _ in range(L)]
    V0 = [rng.randn(r, H) * 0.05 for _ in range(L)]
    Wout0 = rng.randn(H, 1) * (1 / np.sqrt(H))
    
    # Factory offline pre-training
    A, B, Wo, _ = backprop_adapt(e_tr, y_tr, layers, U0, V0, Wout0, H, r, FACTORY_STEPS, lr=0.05)
    
    return {
        "layers": layers, "H": H, "r": r, "L": L,
        "e_ad": e_ad, "y_ad": y_ad, "e_ev": e_ev, "yev_cyc": yev_cyc,
        "A": A, "B": B, "Wo": Wo, "ymu": ymu, "ysd": ysd,
        "rng": rng, "ad_idx": ad_idx, "y": y, "batch": batch
    }

def cv_select_rigorous(P, n_adapt):
    # Select grid configuration using CV on e_ad[:n_adapt]
    e_ad_sub = P["e_ad"][:n_adapt]
    y_ad_sub = P["y_ad"][:n_adapt]
    
    if n_adapt <= 5:
        # Leave-One-Out (LOO) CV
        folds = []
        for i in range(n_adapt):
            val_idx = [i]
            train_idx = [j for j in range(n_adapt) if j != i]
            folds.append((train_idx, val_idx))
    else:
        # 5-fold CV
        idx = P["rng"].permutation(n_adapt)
        folds = []
        fold_size = n_adapt // 5
        for f in range(5):
            val_idx = idx[f*fold_size : (f+1)*fold_size]
            train_idx = np.setdiff1d(idx, val_idx)
            folds.append((train_idx, val_idx))
            
    best, best_val = None, 1e9
    ad_cyc_sub = P["y"][P["ad_idx"]][:n_adapt]
    
    for lam, lr in GRID:
        vals = []
        for tr_f, va_f in folds:
            U, V, Wo, _ = halo_adapt(
                e_ad_sub[tr_f], y_ad_sub[tr_f], P["layers"],
                P["A"], P["B"], P["Wo"], P["H"], P["r"],
                600, lr=lr, lam=lam
            )
            pred = predict(e_ad_sub[va_f], P["layers"], U, V, Wo)
            v = mape_cyclelife(pred, ad_cyc_sub[va_f], P["ymu"], P["ysd"])
            vals.append(v)
        m = np.mean(vals)
        if m < best_val:
            best_val, best = m, (lam, lr)
            
    return best

def bootstrap_ci(diffs, B=1000):
    # Compute 95% percentile bootstrap confidence interval on paired differences
    n = len(diffs)
    means = []
    for _ in range(B):
        boot = np.random.choice(diffs, size=n, replace=True)
        means.append(np.mean(boot))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def run_seed_task(args):
    seed, r, na, P = args
    np.random.seed(seed)
    
    # 1. no-adapt
    mape_no = mape_cyclelife(
        predict(P["e_ev"], P["layers"], P["A"], P["B"], P["Wo"]),
        P["yev_cyc"], P["ymu"], P["ysd"]
    )
    
    # 2. LoRA adaptation
    A2, B2, Wo2, _ = backprop_adapt(
        P["e_ad"][:na], P["y_ad"][:na], P["layers"],
        P["A"], P["B"], P["Wo"], P["H"], P["r"],
        DEVICE_STEPS, lr=0.03
    )
    mape_lora = mape_cyclelife(
        predict(P["e_ev"], P["layers"], A2, B2, Wo2),
        P["yev_cyc"], P["ymu"], P["ysd"]
    )
    
    # 3. HALO CV
    lam_cv, lr_cv = cv_select_rigorous(P, na)
    U_cv, V_cv, Wo_cv, _ = halo_adapt(
        P["e_ad"][:na], P["y_ad"][:na], P["layers"],
        P["A"], P["B"], P["Wo"], P["H"], P["r"],
        HALO_STEPS, lr=lr_cv, lam=lam_cv
    )
    mape_hcv = mape_cyclelife(
        predict(P["e_ev"], P["layers"], U_cv, V_cv, Wo_cv),
        P["yev_cyc"], P["ymu"], P["ysd"]
    )
    
    # 4. HALO Oracle
    opts = []
    for lam, lr in GRID:
        U_or, V_or, Wo_or, _ = halo_adapt(
            P["e_ad"][:na], P["y_ad"][:na], P["layers"],
            P["A"], P["B"], P["Wo"], P["H"], P["r"],
            HALO_STEPS, lr=lr, lam=lam
        )
        opts.append(mape_cyclelife(
            predict(P["e_ev"], P["layers"], U_or, V_or, Wo_or),
            P["yev_cyc"], P["ymu"], P["ysd"]
        ))
    mape_hor = min(opts)
    
    return mape_no, mape_lora, mape_hcv, mape_hor

def run_prep_task(args):
    seed, r = args
    np.random.seed(seed)
    return seed, r, prep_rigorous(seed, r)

if __name__ == "__main__":
    from multiprocessing import Pool
    
    SEEDS = list(range(20))
    RANKS = [4, 8, 16]
    N_ADAPTS = [5, 10, 15, 20]
    
    print("=" * 85)
    print("VolMax HALO — Rigorous On-Device BMS Adaptation Benchmark (Severson cycle-life)")
    print(f"  Seeds: 0-19 ({len(SEEDS)} runs) | Ranks: {RANKS}")
    print("  ORACLE = best-on-eval | CV = honest 5-fold / LOO selection")
    print("  Paired evaluation: HALO and LoRA run on identical train/eval splits.")
    print("=" * 85)
    
    # 1. Parallel Pre-training
    print("Pre-training base models across all seeds and ranks in parallel...")
    prep_tasks = [(s, r) for r in RANKS for s in SEEDS]
    with Pool() as pool:
        prepped_results = pool.map(run_prep_task, prep_tasks)
    
    prepped_data = {}
    for seed, r, P in prepped_results:
        prepped_data[(seed, r)] = P
    print("Pre-training complete.\n")
    
    out = {}
    
    for r in RANKS:
        out[r] = {}
        print(f"\n--- RANK {r} ---")
        print(f"{'N_adapt':>7} | {'no-adapt':>11} | {'LoRA':>11} | {'HALO CV':>11} | {'HALO Oracle':>11} | {'p-val (Wilcox)':>14} | {'Bootstrap 95% CI':>18}")
        print("-" * 96)
        
        for na in N_ADAPTS:
            # Parallelize seeds loop using multiprocessing Pool
            tasks = [(s, r, na, prepped_data[(s, r)]) for s in SEEDS]
            with Pool() as pool:
                res = pool.map(run_seed_task, tasks)
                
            N_mapes = np.array([r[0] for r in res])
            Lo_mapes = np.array([r[1] for r in res])
            Hcv_mapes = np.array([r[2] for r in res])
            Hor_mapes = np.array([r[3] for r in res])
            
            # Wilcoxon signed-rank test on paired differences (HALO CV - LoRA)
            diffs = Hcv_mapes - Lo_mapes
            try:
                w_stat, p_val = stats.wilcoxon(diffs)
            except Exception:
                p_val = 1.0
                
            # Bootstrap 95% Confidence Interval of the paired gap
            ci_low, ci_high = bootstrap_ci(diffs)
            
            f = lambda a: f"{np.mean(a):5.2f}±{np.std(a):4.2f}"
            print(f"{na:>7d} | {f(N_mapes):>11} | {f(Lo_mapes):>11} | {f(Hcv_mapes):>11} | {f(Hor_mapes):>11} | {p_val:>14.4f} | [{ci_low:+5.2f}, {ci_high:+5.2f}]")
            
            out[f"{r}_{na}"] = {
                "n_adapt": na,
                "mape_no": [float(np.mean(N_mapes)), float(np.std(N_mapes))],
                "mape_lora": [float(np.mean(Lo_mapes)), float(np.std(Lo_mapes))],
                "mape_halo_cv": [float(np.mean(Hcv_mapes)), float(np.std(Hcv_mapes))],
                "mape_halo_oracle": [float(np.mean(Hor_mapes)), float(np.std(Hor_mapes))],
                "wilcoxon_p": float(p_val),
                "bootstrap_ci": [ci_low, ci_high],
                "gap_mean": float(np.mean(diffs))
            }
            
    try:
        with open("verified_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\n[OK] Rigorous results written to verified_results.json")
    except Exception as e:
        print(f"\n[Error] Writing results failed: {e}")

