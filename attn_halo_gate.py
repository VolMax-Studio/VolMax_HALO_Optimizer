"""
HALO on attention Q/K/V — CORRECTNESS GATE
===========================================
Frozen INT4 attention block; adapt ONLY Q/K/V projections (+ Wo, head).
Compare: no-adapt | LoRA(backprop) | HALO-DFA | HALO+FOTON | +Precond.
Task: single-head-solvable associative recall (sanity-proven healthy).
Metric: test accuracy. Gate: HALO variants must beat no-adapt and the
ablation ordering must hold (FOTON helps, precond helps), else the
attention-DFA approximation doesn't carry and we report that plainly.
"""
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0)

N_KEYS, N_VALS, N_PAIRS, D = 8, 8, 4, 48
N_PAIR_TOK = N_KEYS*N_VALS
VOCAB = N_PAIR_TOK + N_KEYS
SEQ = N_PAIRS + 1
STEPS = 1500; BATCH = 128; RANK = 8; N_BITS = 4

def make_batch(B):
    seqs,tgts=[],[]
    for _ in range(B):
        keys=np.random.permutation(N_KEYS)[:N_PAIRS]
        vals=np.random.randint(0,N_VALS,size=N_PAIRS)
        toks=[k*N_VALS+v for k,v in zip(keys,vals)]
        qi=np.random.randint(0,N_PAIRS); toks.append(N_PAIR_TOK+keys[qi])
        seqs.append(toks); tgts.append(vals[qi])
    return torch.tensor(seqs), torch.tensor(tgts)

class STE(torch.autograd.Function):
    @staticmethod
    def forward(c,x,nb):
        qm,qM=-(2**(nb-1)),(2**(nb-1)-1); s=x.abs().max().clamp(min=1e-8)/qM
        return torch.clamp(torch.round(x/s),qm,qM)*s
    @staticmethod
    def backward(c,g): return g,None
def fq(x): return STE.apply(x,N_BITS)

def nsmuon(W,it=5):
    if not torch.isfinite(W).all(): W=torch.randn(W.shape[0],W.shape[1])*0.05
    a,b,c=3.4445,-4.7750,2.0315; X=W/(W.norm()+1e-8)
    for _ in range(it): A=X.t()@X; X=a*X+b*(X@A)+c*(X@(A@A))
    return X

# Shared frozen INT4 base: embedding, pos, and a FROZEN INT4 Q/K/V/Wo.
torch.manual_seed(1)
EMB = torch.randn(VOCAB,D)*0.1
POS = torch.randn(SEQ,D)*0.1
Wq0,Wk0,Wv0,Wo0 = [fq(torch.randn(D,D)*(1/np.sqrt(D))) for _ in range(4)]

def embed(x): return EMB[x] + POS[None,:,:]

def attn_forward(h, Wq, Wk, Wv, Wo):
    q,k,v = h@Wq, h@Wk, h@Wv
    att = torch.softmax(q @ k.transpose(1,2)/np.sqrt(D), dim=-1)
    o = (att @ v) @ Wo
    return o, (q,k,v,att)

Xte,yte = make_batch(3000)

def evaluate(headW, adapt=None):
    with torch.no_grad():
        h = embed(Xte)
        if adapt is None:
            o,_ = attn_forward(h,Wq0,Wk0,Wv0,Wo0)
        else:
            o = adapt(h)
        logits = o[:,-1,:] @ headW
        return (logits.argmax(1)==yte).float().mean().item()

# ---- no-adapt floor (frozen QKV, only head trained by delta rule) ----
def run_no_adapt():
    torch.manual_seed(2); headW = torch.randn(D,N_VALS)*(1/np.sqrt(D)); hlr=0.05
    for s in range(STEPS):
        xb,yb=make_batch(BATCH)
        with torch.no_grad():
            h=embed(xb); o,_=attn_forward(h,Wq0,Wk0,Wv0,Wo0)
            feat=o[:,-1,:]; logits=feat@headW
            p=torch.softmax(logits,1); oh=torch.zeros_like(p); oh[torch.arange(BATCH),yb]=1
            headW-=hlr*(feat.t()@(p-oh))/BATCH
    return evaluate(headW)

# ---- LoRA on Q/K/V (+Wo,head), real backprop ----
def run_lora():
    torch.manual_seed(2)
    A={n:nn.Parameter(torch.randn(D,RANK)*0.01) for n in "qkvo"}
    Bk={n:nn.Parameter(torch.zeros(RANK,D)) for n in "qkvo"}
    headW=nn.Parameter(torch.randn(D,N_VALS)*(1/np.sqrt(D)))
    ps=list(A.values())+list(Bk.values())+[headW]
    opt=torch.optim.Adam(ps,lr=3e-3)
    def fwd(h):
        Wq=Wq0+A['q']@Bk['q']; Wk=Wk0+A['k']@Bk['k']; Wv=Wv0+A['v']@Bk['v']; Wo=Wo0+A['o']@Bk['o']
        o,_=attn_forward(h,Wq,Wk,Wv,Wo); return o
    for s in range(STEPS):
        xb,yb=make_batch(BATCH)
        o=fwd(embed(xb)); logits=o[:,-1,:]@headW
        loss=nn.functional.cross_entropy(logits,yb)
        opt.zero_grad(); loss.backward(); opt.step()
    # peak retained bytes: attention graph (att is B×SEQ×SEQ) + activations
    peak=(BATCH*SEQ*SEQ + BATCH*SEQ*D*4)*4
    return evaluate(headW, adapt=lambda h: fwd(h)), peak

# ---- HALO on Q/K/V: DFA error -> per-projection Oja(+FOTON)(+precond) ----
def run_halo(use_foton=True, precond=False, lam=0.1):
    torch.manual_seed(2)
    U={n:torch.randn(D,RANK)*0.05 for n in "qkvo"}
    V={n:torch.randn(RANK,D)*0.05 for n in "qkvo"}
    headW=torch.randn(D,N_VALS)*(1/np.sqrt(D))
    torch.manual_seed(7)
    R={n:torch.randn(N_VALS,D)*(1/np.sqrt(N_VALS)) for n in "qkvo"}
    Ir=torch.eye(RANK); clip=0.5; lr=0.03; hlr=0.03
    def fwd(h):
        Wq=Wq0+U['q']@V['q']; Wk=Wk0+U['k']@V['k']; Wv=Wv0+U['v']@V['v']; Wo=Wo0+U['o']@V['o']
        o,cache=attn_forward(h,Wq,Wk,Wv,Wo); return o,cache,(Wq,Wk,Wv,Wo)
    peak=0
    for s in range(STEPS):
        xb,yb=make_batch(BATCH)
        with torch.no_grad():
            h=embed(xb)
            o,cache,_=fwd(h)
            feat=o[:,-1,:]; logits=feat@headW
            p=torch.softmax(logits,1); oh=torch.zeros_like(p); oh[torch.arange(BATCH),yb]=1
            gerr=p-oh                                  # (B,N_VALS) global task error
            headW-=hlr*(feat.t()@gerr)/BATCH
            # DFA: broadcast global error to each projection's input space.
            # Each projection takes input h (B,SEQ,D); we use the last-token
            # row as the credit-bearing position (that's where readout is).
            h_last = h[:,-1,:]                          # (B,D)
            for n in "qkvo":
                e_local = gerr @ R[n]                   # (B,D) projected task error
                z = h_last @ U[n]                       # (B,RANK)
                dU = (h_last.t()@z)/BATCH - (h_last**2).mean(0).unsqueeze(1)*U[n]
                dV = (z.t()@e_local)/BATCH
                if precond:
                    Gz=(z.t()@z)/BATCH+lam*Ir; Gzi=torch.linalg.inv(Gz)
                    dU=dU@Gzi; dV=Gzi@dV
                U[n]=U[n]+lr*dU.clamp(-clip,clip)
                V[n]=V[n]-lr*dV.clamp(-clip,clip)
                if use_foton: U[n]=nsmuon(U[n])
                cur=(h_last.numel()+z.numel()+e_local.numel()+U[n].numel()+V[n].numel())*4
                peak=max(peak,cur)
    return evaluate(headW, adapt=lambda h: fwd(h)[0]), peak

print("="*78)
print("HALO on attention Q/K/V — correctness gate (associative recall)")
print("="*78)
na = run_no_adapt()
lora,b_lora = run_lora()
dfa,_ = run_halo(use_foton=False, precond=False)
ft,_  = run_halo(use_foton=True,  precond=False)
pc,b_pc = run_halo(use_foton=True, precond=True)
print(f"  no-adapt (frozen QKV)      : {na:.3f}")
print(f"  LoRA on QKV (backprop)     : {lora:.3f}   [peak {b_lora} B]")
print(f"  HALO-DFA (no ortho)        : {dfa:.3f}")
print(f"  HALO+FOTON                 : {ft:.3f}")
print(f"  HALO+FOTON+Precond         : {pc:.3f}   [peak {b_pc} B]")
print(f"  random baseline            : {1/N_VALS:.3f}")

# ---- Localization test: HALO on Wo ONLY (linear), Q/K/V frozen ----
def run_halo_wo_only(precond=True, lam=0.1):
    torch.manual_seed(2)
    U=torch.randn(D,RANK)*0.05; V=torch.randn(RANK,D)*0.05
    headW=torch.randn(D,N_VALS)*(1/np.sqrt(D))
    torch.manual_seed(7); R=torch.randn(N_VALS,D)*(1/np.sqrt(N_VALS))
    Ir=torch.eye(RANK); clip=0.5; lr=0.03; hlr=0.03
    def fwd(h):
        Wo=Wo0+U@V
        o,_=attn_forward(h,Wq0,Wk0,Wv0,Wo); return o
    for s in range(STEPS):
        xb,yb=make_batch(BATCH)
        with torch.no_grad():
            h=embed(xb); o=fwd(h); feat=o[:,-1,:]; logits=feat@headW
            p=torch.softmax(logits,1); oh=torch.zeros_like(p); oh[torch.arange(BATCH),yb]=1
            gerr=p-oh; headW-=hlr*(feat.t()@gerr)/BATCH
            # the input to Wo is (att@v) at last position
            q,k,v=h@Wq0,h@Wk0,h@Wv0
            att=torch.softmax(q@k.transpose(1,2)/np.sqrt(D),dim=-1)
            wo_in=(att@v)[:,-1,:]                       # (B,D) input to Wo
            e_local=gerr@R; z=wo_in@U
            dU=(wo_in.t()@z)/BATCH-(wo_in**2).mean(0).unsqueeze(1)*U
            dV=(z.t()@e_local)/BATCH
            Gz=(z.t()@z)/BATCH+lam*Ir; Gzi=torch.linalg.inv(Gz)
            dU=(dU@Gzi).clamp(-clip,clip); dV=(Gzi@dV).clamp(-clip,clip)
            U=U+lr*dU; V=V-lr*dV; U=nsmuon(U)
    return evaluate(headW, adapt=lambda h: fwd(h))

wo = run_halo_wo_only()
print(f"\n  LOCALIZATION:")
print(f"  HALO on Wo only (linear, QKV frozen): {wo:.3f}")
print(f"  -> if >> no-adapt (0.177): problem is isolated to softmax (Q/K), not linear projections")
print(f"  -> if ~ random: even linear attention-output adaptation fails forward-only")
