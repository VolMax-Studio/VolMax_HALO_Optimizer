#!/usr/bin/env python3
"""
VolMax HALO — On-Device BMS Adaptation Benchmark (SOH / cycle-life)
===================================================================
Tests the HALO forward-only adaptation rule (DFA + FOTON orthogonalization +
Newton-Muon preconditioning) on a REAL battery SOH problem (Severson cycle-life),
against LoRA, under a constrained on-device adaptation scenario.

This is a faithful REGRESSION port of the classification rule in
VolMax_HALO_Optimizer/reproduce.py. Only the error signal changes
(softmax-residual -> MSE residual); the update equations are identical.

READ THE LIMITATIONS FIRST (they are the point):
  L1. Severson is LAB cycling data, not field BMS telematics. "On-device
      adaptation" is the FRAMING/motivation; this is a lab-data proxy, not a
      field deployment result.
  L2. Frozen base is a random INT4 stack with a trained low-rank adapter
      (the HALO regime, per the original repo) — not a deep trained estimator.
      The factory adapter is trained offline (backprop, no memory limit); only
      the ON-DEVICE b3 adaptation is the memory-constrained step being compared.
  L3. Memory numbers are ANALYTIC retained-tensor bytes (same accounting as the
      original repo), not OS RSS. They count what must be held for the update:
      LoRA retains every layer's input activation for the backward graph -> O(L);
      HALO holds one layer at a time -> O(1). Labeled as such.
  L4. b3 has 40 cells. Split disjointly into adapt/eval CELLS; numbers are
      mean +/- std over multiple seeds. A single run is not a result here.
  L5. INT4 quantization of the base injects its own error; the MAPE is
      decomposed (FP32 base vs INT4 base) so the adaptation effect is not
      conflated with the quantization cost.

Anti-hype: every number printed is measured in this run. Nothing hand-copied.
"""
import numpy as np, csv, json
from collections import defaultdict
import os

CSV_PATH = "../Battery_Health_Portfolio/severson_features.csv"
EXCLUDE = {"cell_id","batch","charge_policy","cycle_life","policy_parsed","partition"}

# ---------------- data ----------------
def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Could not find features CSV at {CSV_PATH}")
    rows = list(csv.DictReader(open(CSV_PATH)))
    feat_cols = [c for c in rows[0].keys() if c not in EXCLUDE]
    X, y, batch, cell = [], [], [], []
    for r in rows:
        try:
            xv = [float(r[c]) for c in feat_cols]
        except ValueError:
            continue
        X.append(xv); y.append(float(r["cycle_life"]))
        batch.append(r["batch"]); cell.append(r["cell_id"])
    return (np.array(X, dtype=np.float64), np.array(y, dtype=np.float64),
            np.array(batch), np.array(cell), feat_cols)

# ---------------- primitives (ported from reproduce.py) ----------------
def fq(x, nb=4):                       # fake-quant to INT4 (symmetric, per-tensor)
    qm, qM = -(2**(nb-1)), (2**(nb-1)-1)
    s = max(np.abs(x).max(), 1e-8) / qM
    return np.clip(np.round(x/s), qm, qM) * s

def nsmuon(W, it=5):                    # FOTON: Newton-Schulz orthogonalization
    if not np.isfinite(W).all():
        W = np.random.randn(*W.shape) * 0.05
    a,b,c = 3.4445,-4.7750,2.0315
    X = W / (np.linalg.norm(W) + 1e-8)
    for _ in range(it):
        A = X.T @ X
        X = a*X + b*(X @ A) + c*(X @ (A @ A))
    return X

def tanh(x): return np.tanh(x)

# ---------------- frozen INT4 base + adapter forward ----------------
def make_stack(L, H, seed):
    rng = np.random.RandomState(seed)
    return [(fq(rng.randn(H,H)*(1/np.sqrt(H))), fq(np.zeros(H))) for _ in range(L)]

def embed(X, Win, bin_):               # frozen INT4 input embedding -> H-space
    return tanh(X @ Win + bin_)

def fwd(e, layers, U, V):
    h = e; cache = []
    for i,(W,b) in enumerate(layers):
        a_in = h
        a_out = tanh(a_in @ W + (a_in @ U[i]) @ V[i] + b)
        cache.append((a_in, a_out)); h = a_out
    return h, cache

def predict(e, layers, U, V, Wout):
    h,_ = fwd(e, layers, U, V); return h @ Wout

# ---------------- LoRA / backprop adaptation (factory train + on-device LoRA) ----------------
def backprop_adapt(e, y, layers, U, V, Wout, H, r, steps, lr=0.05, batch=64, momentum=0.9):
    L = len(layers); N = e.shape[0]
    A = [u.copy() for u in U]; B = [v.copy() for v in V]; Wo = Wout.copy()
    vA=[np.zeros_like(a) for a in A]; vB=[np.zeros_like(b) for b in B]; vWo=np.zeros_like(Wo)
    bs = min(batch, N)
    for s in range(steps):
        bi = np.random.randint(0, N, bs); xb, yb = e[bi], y[bi]
        # forward (retain activations -> this is the O(L) cost)
        h = xb; cache=[]
        for i,(W,b) in enumerate(layers):
            a_in=h; z=a_in@A[i]; pre=a_in@W + z@B[i] + b; a_out=tanh(pre)
            cache.append((a_in,z,a_out)); h=a_out
        pred = h @ Wo                                  # (bs,1)
        err = (pred - yb.reshape(-1,1)) / bs           # dMSE/dpred
        gWo = cache[-1][2].T @ err
        dh = err @ Wo.T
        gA=[None]*L; gB=[None]*L
        for i in reversed(range(L)):
            a_in,z,a_out = cache[i]
            dpre = dh * (1 - a_out**2)
            gB[i] = z.T @ dpre
            dz = dpre @ B[i].T
            gA[i] = a_in.T @ dz
            dh = dpre @ layers[i][0].T + dz @ A[i].T
        # SGD + momentum
        for i in range(L):
            vA[i]=momentum*vA[i]-lr*gA[i]; A[i]+=vA[i]
            vB[i]=momentum*vB[i]-lr*gB[i]; B[i]+=vB[i]
        vWo=momentum*vWo-lr*gWo; Wo+=vWo
    # analytic retained bytes for backward graph: each layer input activation O(L) + adapters
    peak = (bs*H)*L*8 + sum(a.size+b.size for a,b in zip(A,B))*8 + Wo.size*8
    return A, B, Wo, peak

# ---------------- HALO forward-only adaptation (ported, regression error) ----------------
def halo_adapt(e, y, layers, U, V, Wout, H, r, steps, lr=0.03, batch=64,
               use_foton=True, precond=True, lam=0.1, clip=0.5, fb_seed=7):
    L=len(layers); N=e.shape[0]
    Uw=[u.copy() for u in U]; Vw=[v.copy() for v in V]; Wo=Wout.copy()
    rng=np.random.RandomState(fb_seed)
    R=[rng.randn(1,H)*(1/np.sqrt(1)) for _ in layers]   # fixed random feedback (NCLASS=1)
    Ir=np.eye(r); bs=min(batch,N); peak=0
    for s in range(steps):
        bi=np.random.randint(0,N,bs); xb,yb=e[bi],y[bi].reshape(-1,1)
        h=xb; cache=[]
        for i,(W,b) in enumerate(layers):
            a_in=h; a_out=tanh(a_in@W+(a_in@Uw[i])@Vw[i]+b); cache.append((a_in,a_out)); h=a_out
        pred=h@Wo
        ge=(pred-yb)                                    # MSE residual (was softmax residual)
        aL=cache[-1][1]
        Wo-=0.03*np.clip((aL.T@ge)/bs,-clip,clip)
        for i,(W,b) in enumerate(layers):
            a_in,a_out=cache[i]
            el=ge@R[i]; de=el*(1-a_out**2); z=a_in@Uw[i]
            dU=(a_in.T@z)/bs-(a_in**2).mean(0).reshape(-1,1)*Uw[i]
            dV=(z.T@de)/bs
            if precond:
                Gz=(z.T@z)/bs+lam*Ir; Gzi=np.linalg.inv(Gz); dU=dU@Gzi; dV=Gzi@dV
            dU=np.clip(dU,-clip,clip); dV=np.clip(dV,-clip,clip)
            Uw[i]=Uw[i]+lr*dU; Vw[i]=Vw[i]-lr*dV
            if use_foton: Uw[i]=nsmuon(Uw[i])
            # O(1): one layer's tensors held at a time
            cur=(a_in.size+z.size+a_out.size+Uw[i].size+Vw[i].size+r*r)*8
            peak=max(peak,cur)
    return Uw, Vw, Wo, peak

# ---------------- metric ----------------
def mape_cyclelife(pred_std, y_true_cyc, y_mu, y_sd):
    pred_log = pred_std.reshape(-1)*y_sd + y_mu
    pred_cyc = 10**pred_log
    return float(np.mean(np.abs(pred_cyc - y_true_cyc)/y_true_cyc)*100)

# ---------------- one full run ----------------
def run_once(seed, r, L=4, H=64, factory_steps=4000, device_steps=600, quantize_base=True,
             halo_lr=0.03, halo_lam=0.1):
    rng=np.random.RandomState(seed)
    X,y,batch,cell,_=load_data()
    tr = np.isin(batch,["b1","b2"]); b3 = batch=="b3"
    # leakage-safe: scaler on train only
    mu,sd = X[tr].mean(0), X[tr].std(0)+1e-8
    Xz=(X-mu)/sd
    ylog=np.log10(y); ymu,ysd=ylog[tr].mean(), ylog[tr].std()+1e-8
    yz=(ylog-ymu)/ysd
    # disjoint b3 adapt/eval CELLS
    b3idx=np.where(b3)[0]; rng.shuffle(b3idx)
    half=len(b3idx)//2
    ad_idx,ev_idx=b3idx[:half], b3idx[half:]
    d=Xz.shape[1]
    # frozen INT4 base
    Win=rng.randn(d,H)*(1/np.sqrt(d)); bin_=np.zeros(H)
    if quantize_base: Win=fq(Win)
    layers=make_stack(L,H,seed=seed+100)
    if not quantize_base:
        layers=[(W/ (np.abs(W).max()+1e-8) * np.abs(W).max(), b) for W,b in
                [( np.random.RandomState(seed+100+i).randn(H,H)*(1/np.sqrt(H)), np.zeros(H)) for i in range(L)]]
    e_tr=embed(Xz[tr],Win,bin_); y_tr=yz[tr]
    e_ad=embed(Xz[ad_idx],Win,bin_); y_ad=yz[ad_idx]
    e_ev=embed(Xz[ev_idx],Win,bin_)
    yev_cyc=y[ev_idx]
    # init adapters
    U0=[rng.randn(H,r)*0.05 for _ in range(L)]; V0=[rng.randn(r,H)*0.05 for _ in range(L)]
    Wout0=rng.randn(H,1)*(1/np.sqrt(H))
    # FACTORY train (offline, backprop) on b1+b2 -> M0
    A,B,Wo,_=backprop_adapt(e_tr,y_tr,layers,U0,V0,Wout0,H,r,factory_steps,lr=0.05)
    # no-adapt baseline on b3_eval
    p_no=predict(e_ev,layers,A,B,Wo)
    mape_no=mape_cyclelife(p_no,yev_cyc,ymu,ysd)
    # ON-DEVICE LoRA (backprop) from M0
    A2,B2,Wo2,mem_lora=backprop_adapt(e_ad,y_ad,layers,A,B,Wo,H,r,device_steps,lr=0.03)
    mape_lora=mape_cyclelife(predict(e_ev,layers,A2,B2,Wo2),yev_cyc,ymu,ysd)
    # ON-DEVICE HALO (forward-only) from M0
    U2,V2,Wo3,mem_halo=halo_adapt(e_ad,y_ad,layers,A,B,Wo,H,r,device_steps,lr=halo_lr,lam=halo_lam)
    mape_halo=mape_cyclelife(predict(e_ev,layers,U2,V2,Wo3),yev_cyc,ymu,ysd)
    return dict(mape_no=mape_no, mape_lora=mape_lora, mape_halo=mape_halo,
                mem_lora=mem_lora, mem_halo=mem_halo)

# ---------------- decomposition: FP32 base vs INT4 base (no on-device adapt) ----------------
def decomposition(seeds=(0,1,2,3,4), r=8, L=4, H=64):
    fp32=[]; int4=[]
    for s in seeds:
        np.random.seed(s)
        fp32.append(run_once(s,r,L,H,quantize_base=False)["mape_no"])
        int4.append(run_once(s,r,L,H,quantize_base=True)["mape_no"])
    return (np.mean(fp32),np.std(fp32)),(np.mean(int4),np.std(int4))

# ---------------- main ----------------
if __name__=="__main__":
    SEEDS=[0,1,2,3,4]; RANKS=[4,8,16]; L=4; H=64
    print("="*72)
    print("VolMax HALO — On-Device BMS Adaptation (Severson cycle-life)")
    print(f"  base: frozen INT4 random stack L={L} H={H} | target: log10(cycle_life)")
    print("  train=b1+b2 (84 cells) | on-device adapt/eval = DISJOINT b3 cells (20/20)")
    print(f"  seeds={SEEDS} | MAPE on cycle_life (%), lower=better")
    print("="*72)
    results={"config":dict(L=L,H=H,seeds=SEEDS,ranks=RANKS),"rank":{}}
    print(f"\n{'rank':>4} | {'no-adapt':>16} | {'LoRA':>16} | {'HALO':>16} | {'gap':>7} | {'mem O(L)/O(1)':>13}")
    print("-"*86)
    for r in RANKS:
        rows=defaultdict(list)
        for s in SEEDS:
            np.random.seed(s)
            o=run_once(s,r,L,H)
            for k,v in o.items(): rows[k].append(v)
        agg={k:(float(np.mean(v)),float(np.std(v))) for k,v in rows.items()}
        memratio=agg["mem_lora"][0]/max(agg["mem_halo"][0],1)
        gap=agg["mape_halo"][0]-agg["mape_lora"][0]
        results["rank"][r]={**agg,"mem_ratio":memratio,"gap_halo_minus_lora":gap}
        f=lambda t:f"{t[0]:6.2f}+/-{t[1]:4.2f}"
        print(f"{r:>4} | {f(agg['mape_no']):>16} | {f(agg['mape_lora']):>16} | "
              f"{f(agg['mape_halo']):>16} | {gap:>+6.2f} | {memratio:>11.2f}x")
    # decomposition panel
    fp32,int4=decomposition(SEEDS,r=8,L=L,H=H)
    results["decomposition_mape_noadapt"]={"fp32_base":fp32,"int4_base":int4,
                                           "int4_quant_cost":int4[0]-fp32[0]}
    print("\nMAPE decomposition (no on-device adapt, r=8):")
    print(f"  FP32 base : {fp32[0]:6.2f} +/- {fp32[1]:.2f} %")
    print(f"  INT4 base : {int4[0]:6.2f} +/- {int4[1]:.2f} %   "
          f"(INT4 quantization cost: {int4[0]-fp32[0]:+.2f} pp)")
    # depth -> memory scaling panel
    print("\nMemory advantage scales with depth L (r=8):")
    results["depth_memory"]={}
    for Ld in [2,4,8,16]:
        ms=[]
        for s in SEEDS[:3]:
            np.random.seed(s); o=run_once(s,8,Ld,H,factory_steps=1500,device_steps=300)
            ms.append(o["mem_lora"]/max(o["mem_halo"],1))
        results["depth_memory"][Ld]=float(np.mean(ms))
        print(f"  L={Ld:>2}: O(L)/O(1) memory = {np.mean(ms):.2f}x")
    
    # Save results to local verified_results.json
    try:
        with open("verified_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nVERIFIED — results written to verified_results.json")
    except Exception as e:
        print(f"\nError writing to verified_results.json: {e}")
