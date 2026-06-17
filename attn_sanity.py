"""
Associative recall — SANITY v2 (single-head solvable)
Each (k,v) pair is ONE combined token. Query position attends to the
pair whose key-part matches, reads its value-part. Single-head solvable.
"""
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0)

N_KEYS, N_VALS, N_PAIRS, D_MODEL = 8, 8, 4, 48
N_PAIR_TOK = N_KEYS * N_VALS
VOCAB = N_PAIR_TOK + N_KEYS          # pair tokens + query-key tokens
SEQ_LEN = N_PAIRS + 1

def make_batch(B):
    seqs, tgts = [], []
    for _ in range(B):
        keys = np.random.permutation(N_KEYS)[:N_PAIRS]
        vals = np.random.randint(0, N_VALS, size=N_PAIRS)
        toks = [k*N_VALS + v for k,v in zip(keys,vals)]   # combined pair tokens
        qi = np.random.randint(0, N_PAIRS)
        toks.append(N_PAIR_TOK + keys[qi])                # query-key token
        seqs.append(toks); tgts.append(vals[qi])
    return torch.tensor(seqs), torch.tensor(tgts)

class TinyAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, D_MODEL)*0.1)
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.head = nn.Linear(D_MODEL, N_VALS)
    def forward(self, x):
        h = self.emb(x) + self.pos[None,:,:]
        q,k,v = self.Wq(h), self.Wk(h), self.Wv(h)
        att = torch.softmax(q @ k.transpose(1,2) / np.sqrt(D_MODEL), dim=-1)
        o = self.Wo(att @ v)
        return self.head(o[:, -1, :])

m = TinyAttn()
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
Xte, yte = make_batch(2000)
print("Full-backprop sanity v2 (single-head solvable associative recall):")
for step in range(1, 1501):
    xb, yb = make_batch(128)
    loss = nn.functional.cross_entropy(m(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        with torch.no_grad():
            acc = (m(Xte).argmax(1) == yte).float().mean().item()
        print(f"  step {step:4d} | loss {loss.item():.3f} | test acc {acc:.3f}")
