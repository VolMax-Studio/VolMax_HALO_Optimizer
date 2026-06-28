#!/usr/bin/env python3
"""
HALO per-rank tuning sweep — is the gap to LoRA inherent or undertuning?
========================================================================
Answers one question honestly: does tuning HALO's hyperparameters per rank
close the accuracy gap to LoRA seen in bess_soh_halo_adaptation.py?

Two numbers, because the difference matters:
  ORACLE  : best HALO config selected ON b3_eval. This is an OPTIMISTIC UPPER
            BOUND — it peeks at the test set. NOT a deployable number. It only
            answers "can ANY tuning close the gap in principle?"
  CV       : best HALO config selected by 2-fold CV WITHIN b3_adapt (the data
            you are actually allowed to see on-device). b3_eval stays pristine.
            This is the honest, deployable number.

Verdict logic:
  oracle closes gap & CV closes gap   -> it was just undertuning.
  oracle closes gap & CV does NOT     -> gap is tuning-sensitive but not reliably
                                         closable on this little data.
  neither closes gap                  -> inherent boundary (document it, like the
                                         attention negative result in the repo).
"""
import numpy as np
import json
from bess_soh_halo_adaptation import (load_data, fq, embed, make_stack,
    backprop_adapt, halo_adapt, predict, mape_cyclelife)

GRID = [(lam, lr) for lam in (0.03, 0.1, 0.3) for lr in (0.03, 0.1)]  # 6 configs
HALO_STEPS = 1200

def prep(seed, r, L=4, H=64, factory_steps=4000):
    rng = np.random.RandomState(seed)
    X, y, batch, cell, _ = load_data()
    tr = np.isin(batch, ["b1", "b2"]); b3 = batch == "b3"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xz = (X - mu) / sd
    ylog = np.log10(y); ymu, ysd = ylog[tr].mean(), ylog[tr].std() + 1e-8
    yz = (ylog - ymu) / ysd
    b3idx = np.where(b3)[0]; rng.shuffle(b3idx); half = len(b3idx)//2
    ad_idx, ev_idx = b3idx[:half], b3idx[half:]
    d = Xz.shape[1]
    Win = fq(rng.randn(d, H)*(1/np.sqrt(d))); bin_ = np.zeros(H)
    layers = make_stack(L, H, seed=seed+100)
    e_tr = embed(Xz[tr], Win, bin_); y_tr = yz[tr]
    e_ad = embed(Xz[ad_idx], Win, bin_); y_ad = yz[ad_idx]
    e_ev = embed(Xz[ev_idx], Win, bin_); yev_cyc = y[ev_idx]
    U0 = [rng.randn(H, r)*0.05 for _ in range(L)]; V0 = [rng.randn(r, H)*0.05 for _ in range(L)]
    Wout0 = rng.randn(H, 1)*(1/np.sqrt(H))
    A, B, Wo, _ = backprop_adapt(e_tr, y_tr, layers, U0, V0, Wout0, H, r, factory_steps, lr=0.05)
    return dict(layers=layers, H=H, r=r, e_ad=e_ad, y_ad=y_ad, e_ev=e_ev,
                yev_cyc=yev_cyc, A=A, B=B, Wo=Wo, ymu=ymu, ysd=ysd, rng=rng,
                ad_idx=ad_idx, X=X, y=y, batch=batch)

def halo_eval(P, lam, lr, e_adapt, y_adapt):
    U, V, Wo, _ = halo_adapt(e_adapt, y_adapt, P["layers"], P["A"], P["B"], P["Wo"],
                             P["H"], P["r"], HALO_STEPS, lr=lr, lam=lam)
    return mape_cyclelife(predict(P["e_ev"], P["layers"], U, V, Wo), P["yev_cyc"], P["ymu"], P["ysd"])

def cv_select(P):
    # 2-fold CV within b3_adapt (by row=cell). pick config by mean val MAPE. eval untouched.
    n = P["e_ad"].shape[0]; idx = P["rng"].permutation(n); h = n//2
    folds = [(idx[:h], idx[h:]), (idx[h:], idx[:h])]
    best, best_val = None, 1e9
    for lam, lr in GRID:
        vals = []
        for tr_f, va_f in folds:
            U, V, Wo, _ = halo_adapt(P["e_ad"][tr_f], P["y_ad"][tr_f], P["layers"],
                                     P["A"], P["B"], P["Wo"], P["H"], P["r"], HALO_STEPS, lr=lr, lam=lam)
            # validate on held-out adapt fold (cycle-life MAPE)
            pred = predict(P["e_ad"][va_f], P["layers"], U, V, Wo)
            # map va_f (positions within ad_idx) back to true cycle-life
            ad_cyc = P["y"][P["ad_idx"]]
            v = mape_cyclelife(pred, ad_cyc[va_f], P["ymu"], P["ysd"])
            vals.append(v)
        m = np.mean(vals)
        if m < best_val: best_val, best = m, (lam, lr)
    return best

if __name__ == "__main__":
    SEEDS = [0, 1, 2, 3, 4]; RANKS = [4, 8, 16]
    print("="*78)
    print("HALO per-rank tuning — gap inherent or undertuning?")
    print(f"  grid lam{{0.03,0.1,0.3}} x lr{{0.03,0.1}}  steps={HALO_STEPS}  seeds={SEEDS}")
    print("  ORACLE = best-on-eval (optimistic upper bound) | CV = honest, eval clean")
    print("="*78)
    print(f"\n{'rank':>4} | {'no-adapt':>11} | {'LoRA':>11} | {'HALO fixed':>11} | "
          f"{'HALO CV':>11} | {'HALO oracle':>11}")
    print("-"*78)
    out = {}
    for r in RANKS:
        N, Lo, Hf, Hcv, Hor = [], [], [], [], []
        for s in SEEDS:
            np.random.seed(s)
            P = prep(s, r)
            # no-adapt
            N.append(mape_cyclelife(predict(P["e_ev"], P["layers"], P["A"], P["B"], P["Wo"]),
                                    P["yev_cyc"], P["ymu"], P["ysd"]))
            # LoRA fixed
            A2, B2, Wo2, _ = backprop_adapt(P["e_ad"], P["y_ad"], P["layers"], P["A"], P["B"], P["Wo"],
                                            P["H"], P["r"], 600, lr=0.03)
            Lo.append(mape_cyclelife(predict(P["e_ev"], P["layers"], A2, B2, Wo2),
                                     P["yev_cyc"], P["ymu"], P["ysd"]))
            # HALO fixed (the original config)
            Hf.append(halo_eval(P, 0.1, 0.03, P["e_ad"], P["y_ad"]))
            # HALO oracle (best on eval — upper bound)
            Hor.append(min(halo_eval(P, lam, lr, P["e_ad"], P["y_ad"]) for lam, lr in GRID))
            # HALO CV-selected (honest)
            lam_cv, lr_cv = cv_select(P)
            Hcv.append(halo_eval(P, lam_cv, lr_cv, P["e_ad"], P["y_ad"]))
        f = lambda a: f"{np.mean(a):5.2f}±{np.std(a):4.2f}"
        out[r] = dict(no=f(N), lora=f(Lo), halo_fixed=f(Hf), halo_cv=f(Hcv), halo_oracle=f(Hor),
                      gap_cv=float(np.mean(Hcv)-np.mean(Lo)), gap_oracle=float(np.mean(Hor)-np.mean(Lo)))
        print(f"{r:>4} | {f(N):>11} | {f(Lo):>11} | {f(Hf):>11} | {f(Hcv):>11} | {f(Hor):>11}")
    
    try:
        with open("halo_tuning_results.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nResults written to halo_tuning_results.json")
    except Exception as e:
        print(f"\nError writing results json: {e}")

    print("\nGap (HALO - LoRA), pp:  positive = HALO still worse")
    for r in RANKS:
        print(f"  r={r:>2}: CV-selected {out[r]['gap_cv']:+.2f}   oracle {out[r]['gap_oracle']:+.2f}")
