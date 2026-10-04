import numpy as np, pandas as pd

base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
bcols = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med

Xb, med = prep(m[bcols])
TR13 = m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values

def bin_features(X, tr, nb=64):
    Xb_ = np.zeros(X.shape, dtype=np.int8)
    for j in range(X.shape[1]):
        qs = np.quantile(X[tr,j], np.linspace(0,1,nb+1)[1:-1])
        Xb_[:,j] = np.searchsorted(qs, X[:,j])
    return Xb_

def gbm(Xb_, y, tr_idx, rounds=120, lr=0.08, depth=3, min_leaf=150, colfrac=0.7, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    pred = np.zeros(n); pred[tr_idx] = y[tr_idx].mean()
    F = Xb_.shape[1]
    for r in range(rounds):
        resid = y - pred
        feats = rng.choice(F, int(F*colfrac), replace=False)
        def build(rows):
            if len(rows) < min_leaf: return ('leaf', resid[rows].mean())
            best = None; gr = resid[rows]
            for j in feats:
                b = Xb_[rows, j]
                c = np.bincount(b, minlength=64).astype(float)
                h = np.bincount(b, weights=gr, minlength=64)
                cs = np.cumsum(c); gs = np.cumsum(h)
                rc, rg = cs[-1], gs[-1]
                if rc < 2*min_leaf: continue
                cl = cs[:-1]; gl = gs[:-1]; cr = rc-cl; grr = rg-gl
                valid = (cl>=min_leaf)&(cr>=min_leaf)
                if not valid.any(): continue
                with np.errstate(invalid='ignore'):
                    score = np.where(valid, gl*gl/np.maximum(cl,1) + grr*grr/np.maximum(cr,1), -np.inf)
                k = int(np.argmax(score))
                if best is None or score[k] > best[0]: best = (score[k], j, k)
            if best is None: return ('leaf', gr.mean())
            _, j, k = best
            mask = Xb_[rows,j] <= k
            return ('node', j, k, build(rows[mask]), build(rows[~mask]))
        tree = build(tr_idx)
        def apply(t, rows):
            if t[0]=='leaf': pred[rows] += lr*t[1]; return
            _, j, k, L, R = t
            b = Xb_[rows, j]
            apply(L, rows[b<=k]); apply(R, rows[b>k])
        apply(tree, np.arange(n))
    return pred

Xbin = bin_features(Xb, TR13)
tr_t = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403]).values)[0]
va_t = np.where((m.snapshot_day==431).values)[0]
p = gbm(Xbin, y, tr_t)
print("GBM proxy temporal (val=431) MAE:", round(float(np.mean(np.abs(y[va_t]-p[va_t]))),3))
rng = np.random.RandomState(0)
va_r = np.where(TR13 & (rng.rand(len(y))<0.25))[0]
tr_r = np.where(TR13 & ~(rng.rand(len(y))<0.25))[0]
p2 = gbm(Xbin, y, tr_r)
print("GBM proxy random MAE:", round(float(np.mean(np.abs(y[va_r]-p2[va_r]))),3))