#!/usr/bin/env python3
"""
VolMax HALO — Rigorous On-Device BMS Adaptation Benchmark (KIT NMC SOH)
======================================================================
Runs the identical validation protocol as Severson on the KIT dataset:
  - 20 seeds (0-19) for variance control.
  - Reports MAPE, RMSE (cycles), and R2.
"""
import numpy as np
import pandas as pd
import json
import os
import scipy.stats as stats
from multiprocessing import Pool
from bess_soh_halo_adaptation import (fq, embed, make_stack,
    backprop_adapt, halo_adapt, predict)

GRID = [(lam, lr) for lam in (0.03, 0.1, 0.3) for lr in (0.03, 0.1)]
HALO_STEPS = 1200
FACTORY_STEPS = 4000
DEVICE_STEPS = 600

FEATURE_FILE = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/VolMax_HALO_Optimizer/kit_features.csv"

def get_rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true)**2)))

def get_r2(pred, true):
    sse = np.sum((true - pred)**2)
    sst = np.sum((true - np.mean(true))**2)
    if sst < 1e-8:
        return 0.0
    return float(1.0 - sse/sst)

def decode_prediction(pred_std, y_mu, y_sd):
    pred_log = pred_std.reshape(-1) * y_sd + y_mu
    return 10**pred_log

def load_kit_data():
    df = pd.read_csv(FEATURE_FILE)
    feat_cols = ["cap_initial", "cap_diff", "cap_slope", "temp_avg", "temp_slope", "r0_initial", "r0_diff", "r0_slope"]
    X = df[feat_cols].values
    y = df["target_cycle_life"].values
    instance = df["instance"].values
    return X, y, instance

def prep_rigorous_kit(seed, r, L=4, H=64):
    rng = np.random.RandomState(seed)
    X, y, instance = load_kit_data()
    
    # Train is instance 1 & 2; OOD (Test) is instance 3
    tr = np.isin(instance, [1, 2])
    te = instance == 3
    
    # Scaler fit only on Train
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xz = (X - mu) / sd
    
    ylog = np.log10(y)
    ymu, ysd = ylog[tr].mean(), ylog[tr].std() + 1e-8
    yz = (ylog - ymu) / ysd
    
    # Deterministic OOD split (20 adapt, 28 eval from the 48 instance 3 cells)
    te_idx = np.where(te)[0]
    rng.shuffle(te_idx)
    ad_idx, ev_idx = te_idx[:20], te_idx[20:]
    
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
    
    # Factory pre-training on Train instances
    A, B, Wo, _ = backprop_adapt(e_tr, y_tr, layers, U0, V0, Wout0, H, r, FACTORY_STEPS, lr=0.05)
    
    return {
        "layers": layers, "H": H, "r": r, "L": L,
        "e_ad": e_ad, "y_ad": y_ad, "e_ev": e_ev, "yev_cyc": yev_cyc,
        "A": A, "B": B, "Wo": Wo, "ymu": ymu, "ysd": ysd,
        "rng": rng, "ad_idx": ad_idx, "y": y
    }

def cv_select_rigorous_kit(P, n_adapt):
    e_ad_sub = P["e_ad"][:n_adapt]
    y_ad_sub = P["y_ad"][:n_adapt]
    
    if n_adapt <= 5:
        folds = []
        for i in range(n_adapt):
            val_idx = [i]
            train_idx = [j for j in range(n_adapt) if j != i]
            folds.append((train_idx, val_idx))
    else:
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
            pred_cyc = decode_prediction(pred, P["ymu"], P["ysd"])
            # Evaluate using MAPE
            v = float(np.mean(np.abs(pred_cyc - ad_cyc_sub[va_f]) / ad_cyc_sub[va_f]) * 100)
            vals.append(v)
        m = np.mean(vals)
        if m < best_val:
            best_val, best = m, (lam, lr)
            
    return best

def bootstrap_ci(diffs, B=1000):
    n = len(diffs)
    means = []
    for _ in range(B):
        boot = np.random.choice(diffs, size=n, replace=True)
        means.append(np.mean(boot))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def run_seed_task(args):
    seed, r, na, P = args
    np.random.seed(seed)
    
    y_true = P["yev_cyc"]
    
    # 1. no-adapt
    pred_no = predict(P["e_ev"], P["layers"], P["A"], P["B"], P["Wo"])
    cyc_no = decode_prediction(pred_no, P["ymu"], P["ysd"])
    mape_no = float(np.mean(np.abs(cyc_no - y_true)/y_true)*100)
    rmse_no = get_rmse(cyc_no, y_true)
    r2_no = get_r2(cyc_no, y_true)
    
    # 2. LoRA adaptation
    A2, B2, Wo2, _ = backprop_adapt(
        P["e_ad"][:na], P["y_ad"][:na], P["layers"],
        P["A"], P["B"], P["Wo"], P["H"], P["r"],
        DEVICE_STEPS, lr=0.03
    )
    pred_lora = predict(P["e_ev"], P["layers"], A2, B2, Wo2)
    cyc_lora = decode_prediction(pred_lora, P["ymu"], P["ysd"])
    mape_lora = float(np.mean(np.abs(cyc_lora - y_true)/y_true)*100)
    rmse_lora = get_rmse(cyc_lora, y_true)
    r2_lora = get_r2(cyc_lora, y_true)
    
    # 3. HALO CV
    lam_cv, lr_cv = cv_select_rigorous_kit(P, na)
    U_cv, V_cv, Wo_cv, _ = halo_adapt(
        P["e_ad"][:na], P["y_ad"][:na], P["layers"],
        P["A"], P["B"], P["Wo"], P["H"], P["r"],
        HALO_STEPS, lr=lr_cv, lam=lam_cv
    )
    pred_hcv = predict(P["e_ev"], P["layers"], U_cv, V_cv, Wo_cv)
    cyc_hcv = decode_prediction(pred_hcv, P["ymu"], P["ysd"])
    mape_hcv = float(np.mean(np.abs(cyc_hcv - y_true)/y_true)*100)
    rmse_hcv = get_rmse(cyc_hcv, y_true)
    r2_hcv = get_r2(cyc_hcv, y_true)
    
    # 4. HALO Oracle
    opts_mape = []
    opts_rmse = []
    opts_r2 = []
    for lam, lr in GRID:
        U_or, V_or, Wo_or, _ = halo_adapt(
            P["e_ad"][:na], P["y_ad"][:na], P["layers"],
            P["A"], P["B"], P["Wo"], P["H"], P["r"],
            HALO_STEPS, lr=lr, lam=lam
        )
        pred_or = predict(P["e_ev"], P["layers"], U_or, V_or, Wo_or)
        cyc_or = decode_prediction(pred_or, P["ymu"], P["ysd"])
        opts_mape.append(float(np.mean(np.abs(cyc_or - y_true)/y_true)*100))
        opts_rmse.append(get_rmse(cyc_or, y_true))
        opts_r2.append(get_r2(cyc_or, y_true))
        
    best_idx = np.argmin(opts_mape)
    mape_hor = opts_mape[best_idx]
    rmse_hor = opts_rmse[best_idx]
    r2_hor = opts_r2[best_idx]
    
    return {
        "no": (mape_no, rmse_no, r2_no),
        "lora": (mape_lora, rmse_lora, r2_lora),
        "hcv": (mape_hcv, rmse_hcv, r2_hcv),
        "hor": (mape_hor, rmse_hor, r2_hor)
    }

def run_prep_task(args):
    seed, r = args
    np.random.seed(seed)
    return seed, r, prep_rigorous_kit(seed, r)

if __name__ == "__main__":
    SEEDS = list(range(20))
    RANKS = [4, 8, 16]
    N_ADAPTS = [5, 10, 15, 20]
    
    print("=" * 85)
    print("VolMax HALO — Rigorous On-Device BMS Adaptation Benchmark (KIT NMC SOH, Option C)")
    print(f"  Seeds: 0-19 ({len(SEEDS)} runs) | Ranks: {RANKS}")
    print("  ORACLE = best-on-eval | CV = honest 5-fold / LOO selection")
    print("  Split: Instance split (Instances 1 & 2 Train -> Instance 3 OOD)")
    print("=" * 85)
    
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
        print(f"{'N_adapt':>7} | {'no-adapt (M/R/R2)':>22} | {'LoRA (M/R/R2)':>22} | {'HALO CV (M/R/R2)':>22} | {'p-val':>8} | {'Bootstrap 95% CI':>18}")
        print("-" * 110)
        
        for na in N_ADAPTS:
            tasks = [(s, r, na, prepped_data[(s, r)]) for s in SEEDS]
            with Pool() as pool:
                res = pool.map(run_seed_task, tasks)
                
            N_mapes = np.array([r["no"][0] for r in res])
            N_rmses = np.array([r["no"][1] for r in res])
            N_r2s = np.array([r["no"][2] for r in res])
            
            Lo_mapes = np.array([r["lora"][0] for r in res])
            Lo_rmses = np.array([r["lora"][1] for r in res])
            Lo_r2s = np.array([r["lora"][2] for r in res])
            
            Hcv_mapes = np.array([r["hcv"][0] for r in res])
            Hcv_rmses = np.array([r["hcv"][1] for r in res])
            Hcv_r2s = np.array([r["hcv"][2] for r in res])
            
            Hor_mapes = np.array([r["hor"][0] for r in res])
            Hor_rmses = np.array([r["hor"][1] for r in res])
            Hor_r2s = np.array([r["hor"][2] for r in res])
            
            diffs = Hcv_mapes - Lo_mapes
            try:
                w_stat, p_val = stats.wilcoxon(diffs)
            except Exception:
                p_val = 1.0
                
            ci_low, ci_high = bootstrap_ci(diffs)
            
            f = lambda m, rm, r2: f"{np.mean(m):4.1f}%/{np.mean(rm):3.0f}c/{np.mean(r2):.2f}"
            print(f"{na:>7d} | {f(N_mapes, N_rmses, N_r2s):>22} | {f(Lo_mapes, Lo_rmses, Lo_r2s):>22} | {f(Hcv_mapes, Hcv_rmses, Hcv_r2s):>22} | {p_val:>8.4f} | [{ci_low:+5.2f}, {ci_high:+5.2f}]")
            
            out[f"{r}_{na}"] = {
                "n_adapt": na,
                "mape_no": [float(np.mean(N_mapes)), float(np.std(N_mapes))],
                "rmse_no": [float(np.mean(N_rmses)), float(np.std(N_rmses))],
                "r2_no": [float(np.mean(N_r2s)), float(np.std(N_r2s))],
                
                "mape_lora": [float(np.mean(Lo_mapes)), float(np.std(Lo_mapes))],
                "rmse_lora": [float(np.mean(Lo_rmses)), float(np.std(Lo_rmses))],
                "r2_lora": [float(np.mean(Lo_r2s)), float(np.std(Lo_r2s))],
                
                "mape_halo_cv": [float(np.mean(Hcv_mapes)), float(np.std(Hcv_mapes))],
                "rmse_halo_cv": [float(np.mean(Hcv_rmses)), float(np.std(Hcv_rmses))],
                "r2_halo_cv": [float(np.mean(Hcv_r2s)), float(np.std(Hcv_r2s))],
                
                "mape_halo_oracle": [float(np.mean(Hor_mapes)), float(np.std(Hor_mapes))],
                "rmse_halo_oracle": [float(np.mean(Hor_rmses)), float(np.std(Hor_rmses))],
                "r2_halo_oracle": [float(np.mean(Hor_r2s)), float(np.std(Hor_r2s))],
                
                "wilcoxon_p": float(p_val),
                "bootstrap_ci": [ci_low, ci_high],
                "gap_mean": float(np.mean(diffs))
            }
            
    output_json = "/home/volmax-studio/volmax-projects/iot2/PORTFOLIO/VolMax_HALO_Optimizer/kit_verified_results.json"
    try:
        with open(output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[OK] Rigorous results written to {output_json}")
    except Exception as e:
        print(f"\n[Error] Writing results failed: {e}")
