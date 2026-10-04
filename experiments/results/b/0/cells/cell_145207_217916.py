import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def make_bins(Xtr, nbin=64):
    edges = []
    for j in range(Xtr.shape[1]):
        qs = np.quantile(Xtr[:,j], np.linspace(0,1,nbin+1)[1:-1])
        edges.append(np.unique(qs))
    def binn(X):
        B = np.zeros(X.shape, dtype=np.int16)
        for j,e in enumerate(edges):
            B[:,j] = np.searchsorted(e, X[:,j], side='right')
        return B
    return binn

def fit_gbm(B, yy, n_trees=400, lr=0.06, max_depth=4, min_leaf=30, rowsample=0.7, colsample=0.8, seed=1):
    rng = np.random.RandomState(seed)
    n, d = B.shape
    F = np.zeros(n); trees = []
    for t in range(n_trees):
        resid = yy - F
        idx = rng.choice(n, int(n*rowsample), replace=False)
        cols = rng.choice(d, int(d*colsample), replace=False)
        tree = build(idx, cols, 0)
        trees.append((tree, cols))
        F += lr * pred_tree(tree, B, cols, np.arange(n))
    return trees

def build(idx, cols, depth, max_depth=4, min_leaf=30):
    if depth>=max_depth or len(idx)<2*min_leaf:
        return (-1, resid[idx].mean())
    tot = resid[idx].sum(); cnt = len(idx)
    base = tot*tot/cnt
    best = None
    for j in cols:
        b = B[idx, j]
        o = np.argsort(b, kind='stable')
        bs = b[o]; rs = resid[idx][o]
        cs = np.cumsum(rs)
        valid = np.zeros(len(idx), bool); valid[1:] = bs[1:]!=bs[:-1]
        pos = np.where(valid)[0]
        if len(pos)==0: continue
        gl = cs[pos]; gr = tot - gl
        nl = (pos+1).astype(float); nr = cnt - nl
        gain = gl*gl/np.maximum(nl,1) + gr*gr/np.maximum(nr,1) - base
        k = int(np.argmax(gain))
        if gain[k] <= 1e-9: continue
        if best is None or gain[k] > best[0]:
            best = (gain[k], j, bs[pos[k]])
    if best is None:
        return (-1, resid[idx].mean())
    g, j, thr = best
    mask = B[idx, j] <= thr
    L = idx[mask]; R = idx[~mask]
    if len(L)<min_leaf or len(R)<min_leaf:
        return (-1, resid[idx].mean())
    return (j, thr, build(L, cols, depth+1), build(R, cols, depth+1))

def pred_tree(tree, B, idx):
    j, thr, Ls, Rs = tree
    if j == -1:
        return np.full(len(idx), thr)
    mask = B[idx, j] <= thr
    out = np.zeros(len(idx))
    out[mask] = pred_tree(Ls, B, idx[mask])
    out[~mask] = pred_tree(Rs, B, idx[~mask])
    return out

feats = ['spend28','spend56','spend112','trips28','actdays28','nprod28','recency','tenure','trips56','nprod56']
X = np.log1p(np.clip(m[feats].fillna(0).values,0,None))
tr = m.snapshot_day<431; te = m.snapshot_day==431
binn = make_bins(X[tr])
Btr, Bte = binn(X[tr]), binn(X[te])
trees = fit_gbm(Btr, y[tr])
p = np.zeros(int(te.sum()))
for tree, cols in trees:
    p += 0.06 * pred_tree(tree, Bte, np.arange(len(p)))
print('toy GBM 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3))
mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
Xs=(X-mu)/sd; A=Xs[tr].T@Xs[tr]+20*np.eye(len(feats)); w=np.linalg.solve(A,Xs[tr].T@(y[tr]-y[tr].mean()))
pr = Xs[te]@w + y[tr].mean()
print('ridge same feats 431 MAE:', round(float(np.mean(np.abs(y[te]-pr))),3))