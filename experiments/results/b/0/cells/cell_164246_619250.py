import agent_api, pandas as pd, numpy as np
from datetime import datetime as dt
T0 = dt.now()
def ts(msg):
    print(msg, (dt.now()-T0).total_seconds())

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
tr = (df.snapshot_day<=403).values
ts('loaded')

NB = 32; NANB = NB-1
def bin_matrix(df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        xv = x[trm]
        if np.all(np.isnan(xv)):
            B[:,j] = NANB; continue
        e = np.unique(np.quantile(xv, np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b, 0, NANB))
    return B

feats = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
ts('start binning')
B = bin_matrix(df, feats, tr)
ts('binned')

def build_tree(B, rows, r, ci, depth, min_leaf, lam):
    tree = {}
    def rec(R, d):
        if d >= depth or len(R) < 2*min_leaf:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        sub = B[np.ix_(R, ci)]
        idx = sub.astype(np.int32) + NB*np.arange(len(ci))[None,:]
        w = np.repeat(r[R][:,None], len(ci), axis=1).ravel()
        H = np.bincount(idx.ravel(), minlength=NB*len(ci)).reshape(len(ci), NB).astype(float)
        G = np.bincount(idx.ravel(), weights=w, minlength=NB*len(ci)).reshape(len(ci), NB)
        cg = np.cumsum(G, axis=1)[:,:-1]; ch = np.cumsum(H, axis=1)[:,:-1]
        Gt = G.sum(1)[:,None]; Ht = H.sum(1)[:,None]
        gl, gr = cg, Gt-cg; hl, hr = ch, Ht-ch
        gain = gl**2/(hl+lam) + gr**2/(hr+lam) - Gt**2/(Ht+lam)
        gain[hl<min_leaf] = -1; gain[hr<min_leaf] = -1
        fj, bj = np.unravel_index(np.argmax(gain), gain.shape)
        if gain[fj,bj] <= 1e-9:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        f = ci[fj]
        bcol = B[R, f]
        L = R[bcol <= bj]; Rr = R[bcol > bj]
        if len(L)==0 or len(Rr)==0:
            return ('L', lam*r[R].mean()*len(R)/(len(R)+lam))
        nid = len(tree)
        left = rec(L, d+1); right = rec(Rr, d+1)
        tree[nid] = (f, bj, left, right)
        return ('N', nid)
    return rec(rows, 0), tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        f, bj, L, Rr = tree[node[1]]
        bcol = B[rows[idx], f]
        stack.append((L, idx[bcol <= bj])); stack.append((Rr, idx[bcol > bj]))
    return out

y = df.future_spend_4w.values.astype(float)
yw = np.clip(y, 0, np.quantile(y[tr], 0.97))
rng = np.random.RandomState(1)
tr_idx = np.where(tr)[0]
base = yw[tr].mean()
F = np.zeros(len(y))
for t in range(5):
    samp = tr_idx[rng.rand(len(tr_idx)) < 0.6]
    ci = np.arange(len(feats))
    r = yw - F - base
    ts(f'tree {t} start')
    root, tree = build_tree(B, samp, r, ci, 4, 60, 5.0)
    ts(f'tree {t} built, nodes={len(tree)}')
    F += apply_tree(root, tree, B, tr_idx)
    ts(f'tree {t} applied')
print('F stats', F.min(), F.max())
