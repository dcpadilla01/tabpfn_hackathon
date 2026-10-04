import agent_api, pandas as pd, numpy as np

NB = 32; NANB = NB-1

def bin_matrix(Xtr_cols_dict, df, feats, trm):
    B = np.zeros((len(df), len(feats)), dtype=np.uint8)
    edges = []
    for j,c in enumerate(feats):
        x = df[c].values.astype(float)
        e = np.unique(np.quantile(x[trm], np.linspace(0,1,NB)))
        if len(e) <= 2:
            b = (x > e[0]).astype(np.int8)
        else:
            b = np.searchsorted(e[1:-1], x, side='right').astype(np.int8)
        B[:,j] = np.where(np.isnan(x), NANB, np.clip(b, 0, NANB))
        edges.append(e)
    return B, edges

def build_tree(B, rows, r, ci, depth, min_leaf, lam, rng):
    # returns list of nodes: (feat_j, thr_bin, left_rows, right_rows) via recursion
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
        tree[nid] = (f, bj, None, None)
        tree[nid] = (f, bj, rec(L, d+1), rec(Rr, d+1))
        return ('N', nid)
    root = rec(rows, 0)
    return root, tree

def apply_tree(root, tree, B, rows):
    out = np.zeros(len(rows))
    stack = [(root, np.arange(len(rows)))]
    while stack:
        node, idx = stack.pop()
        if node[0]=='L':
            out[idx] = node[1]; continue
        nid = node[1]; f, bj, L, Rr = tree[nid]
        bcol = B[rows[idx], f]
        li = idx[bcol <= bj]; ri = idx[bcol > bj]
        stack.append((L, li)); stack.append((Rr, ri))
    return out

def gbm(df, feats, trm, vam, n_trees=150, lr=0.12, depth=4, min_leaf=50, lam=3.0, clip_q=0.97, seed=1, ncol=40, rsub=0.6, verbose=False):
    y = df.future_spend_4w.values.astype(float)
    yw = np.clip(y, 0, np.quantile(y[trm], clip_q))
    B, edges = bin_matrix(None, df, feats, trm)
    rng = np.random.RandomState(seed)
    tr_idx = np.where(trm)[0]; va_idx = np.where(vam)[0]
    F = np.zeros(len(y)); Fv = np.zeros(len(y))
    base = yw[trm].mean(); F[:] = 0; Fv[:] = 0
    for t in range(n_trees):
        samp = tr_idx[rng.rand(len(tr_idx)) < rsub]
        ci = np.where(rng.rand(len(feats)) < ncol/len(feats))[0]
        if len(ci) < 5: ci = np.arange(len(feats))
        r = yw - F - base
        root, tree = build_tree(B, samp, r, ci, depth, min_leaf, lam, rng)
        Fv += base*0 + 0  # base added once outside
        Fv[va_idx] += apply_tree(root, tree, B, va_idx)
        F[tr_idx] += apply_tree(root, tree, B, tr_idx)
    p = base + Fv[va_idx]
    p = np.clip(p, 0, None)
    return p, y[vam]

df = agent_api.load_saved('e011_price.parquet').merge(agent_api.train_targets(), on=['household_key','snapshot_day'])
core = ['spend28','trips28','recency','spend56','spend112','spend364','tenure','lt_spend']
tr = (df.snapshot_day<=403).values; va = (df.snapshot_day==431).values
p, y = gbm(df, core, tr, va)
print('proxy GBM core8 MAE431:', round(float(np.abs(p-y).mean()),3))
feats = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
p, y = gbm(df, feats, tr, va)
print('proxy GBM all228 MAE431:', round(float(np.abs(p-y).mean()),3))
