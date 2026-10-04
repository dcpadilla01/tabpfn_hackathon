import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()

def prep_matrix(M, tr_mask, nb=64):
    feat = [c for c in M.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
    num_cols = [c for c in feat if M[c].dtype.kind in "ifb"]
    cat_cols = [c for c in feat if M[c].dtype.kind not in "ifb" and M[c].nunique() <= 25]
    Xn = M[num_cols].astype(float).copy()
    med = Xn[tr_mask].median()
    Xn = Xn.fillna(med).fillna(0.0).values
    B = np.empty(Xn.shape, dtype=np.uint8)
    for j in range(Xn.shape[1]):
        qs = np.unique(np.quantile(Xn[tr_mask, j], np.linspace(0,1,nb+1)[1:-1]))
        B[:, j] = np.searchsorted(qs, Xn[:, j], side='right')
    mats = [B]
    for c in cat_cols:
        d = pd.get_dummies(M[c].astype(str), prefix=c, dummy_na=True)
        mats.append(d.values.astype(np.uint8))
    return np.hstack(mats)

def gbm_fit(B, y, snap, tr_snap_max, n_trees=120, lr=0.08, depth=4, nb=64,
            rowsample=0.8, colsample=0.7, min_leaf=20, reg=1.0, seed=0):
    rng = np.random.RandomState(seed)
    trm = snap <= tr_snap_max
    n, k = B.shape
    pred = np.zeros(n)
    tr_idx_all = np.where(trm)[0]
    pred[trm] = y[trm].mean()
    colsel = max(8, int(k*colsample))
    for t in range(n_trees):
        rows = rng.choice(tr_idx_all, size=int(len(tr_idx_all)*rowsample), replace=False)
        feats = rng.choice(k, size=colsel, replace=False)
        res = y - pred
        nodes = [(rows, 0)]
        for d in range(depth):
            new_nodes = []
            for idx, dc, f, b, isleft in nodes:
                if f is None and len(idx) >= 2*min_leaf and dc < depth:
                    G = res[idx].sum(); N = len(idx)
                    best = (0.0, None, None)
                    Bsub = B[idx][:, feats]
                    rsub = res[idx]
                    for fi in range(len(feats)):
                        h = np.bincount(Bsub[:, fi], weights=rsub, minlength=nb)
                        c = np.bincount(Bsub[:, fi], minlength=nb)
                        gl = np.cumsum(h); cl = np.cumsum(c)
                        gr = G - gl; cr = N - cl
                        gain = gl*gl/(cl+reg) + gr*gr/(cr+reg) - G*G/(N+reg)
                        valid = (cl >= min_leaf) & (cr >= min_leaf)
                        gain = np.where(valid, gain, -1e18)
                        bi = int(np.argmax(gain))
                        if gain[bi] > best[0]:
                            best = (gain[bi], fi, bi)
                    if best[1] is None:
                        new_nodes += [(idx, dc, None, None, None)]
                        continue
                    _, fi, bi = best
                    f = feats[fi]
                    mask = B[idx, f] <= bi
                    new_nodes += [(idx[mask], dc+1, f, bi, True),
                                  (idx[~mask], dc+1, f, bi, False)]
                else:
                    new_nodes.append((idx, dc, f, b, isleft))
            nodes = new_nodes
        for idx, dc, f, b, isleft in nodes:
            if f is None and len(idx):
                pred[idx] += lr * res[idx].mean()
    return pred

df = A.load_saved("e015_base.parquet")
M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
trm = (M.snapshot_day <= 375).values
B = prep_matrix(M, trm)
print("B shape", B.shape)
y = np.log1p(M.future_spend_4w.values)
snap = M.snapshot_day.values
pred = gbm_fit(B, y, snap, 375, n_trees=40, lr=0.1, depth=4)
va = snap >= 403
p = np.expm1(pred[va])
print("local MAE (40 trees):", np.mean(np.abs(p - M.future_spend_4w.values[va])))