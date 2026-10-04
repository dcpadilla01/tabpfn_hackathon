
import agent_api, pandas as pd, numpy as np, re
from collections import defaultdict

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
print('E009 shape', t.shape)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
pref = defaultdict(list)
for c in cols:
    p = re.split(r'[\d_]', c)[0]
    pref[p].append(c)
for p in sorted(pref, key=lambda p: -len(pref[p])):
    print(f'{p:22s} {len(pref[p]):4d}  e.g. {pref[p][:5]}')

m = t.merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(m))
y = m['future_spend_4w'].values
print('pct', np.percentile(y,[0,25,50,75,90,95,99]).round(1), 'mean', round(y.mean(),1), 'zero%', round((y==0).mean(),3))
print('MAE pred0', round(np.abs(y).mean(),2))

num = m[cols].select_dtypes(include=[np.number])
print('numeric feats', num.shape[1], 'nonnumeric', len(cols)-num.shape[1])
print('nonnum dtypes:', set(map(str, m[cols].dtypes)))
corr = num.corrwith(m['future_spend_4w'])
cs = corr.abs().sort_values(ascending=False)
print(cs.head(30).round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
print('target by snapshot day (train):')
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()]).round(1).to_string())
# tenure / zero rate by snapshot
print('\nzero% and mean spend_28 by snapshot (all rows):')
print(t.groupby('snapshot_day')[['spend_28','is_zero_28','tenure_days']].mean().round(1).to_string())

# local ridge proxy
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[cols].copy(); Xva = va[cols].copy()
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
for c in cols:
    if str(Xtr[c].dtype) == 'category':
        Xtr[c] = Xtr[c].astype(float); Xva[c] = Xva[c].astype(float)
Xtr = Xtr.astype(np.float64); Xva = Xva.astype(np.float64)
med = np.nanmedian(Xtr.values, axis=0)
Xtr = np.where(np.isnan(Xtr.values), med, Xtr.values)
Xva = np.where(np.isnan(Xva.values), med, Xva.values)
mu = Xtr.mean(0); sg = Xtr.std(0)+1e-9
Xtr_s = (Xtr-mu)/sg; Xva_s = (Xva-mu)/sg

def ridge_fit(X, y, lam):
    A = X.T@X + lam*np.eye(X.shape[1])
    return np.linalg.solve(A, X.T@y)

# pick alpha on inner split: train snapshots <=403 fit, 431 as inner val
inner_va = tr.snapshot_day==431
Xa, ya = Xtr_s[~inner_va.values], ytr[~inner_va.values]
Xb, yb = Xtr_s[inner_va.values], ytr[inner_va.values]
for lam in [1e-2,1e-1,1,10,100,1000]:
    w = ridge_fit(Xa, ya, lam)
    print('lam', lam, 'inner MAE', round(np.abs(Xb@w - yb).mean(),2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def prep(df):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    return X.astype(np.float64).values

Xtr, Xva = prep(tr), prep(va)
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
Xtr_s, Xva_s = (Xtr-mu)/sg, (Xva-mu)/sg

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
Xa, ya, Xb, yb = Xtr_s[~iv], ytr[~iv], Xtr_s[iv], ytr[iv]
print('INNER (431) ridge:')
for lam in [1e-2,1e-1,1,10,100,1000]:
    w = rfit(Xa, ya, lam)
    print(' lam', lam, 'inner MAE', round(np.abs(Xb@w-yb).mean(),2))

print('\nFULL-train fit -> validation MAE (proxy for the harness model):')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    print(' lam', lam, 'val MAE', round(np.abs(Xva_s@w-yva).mean(),2), 'R2', round(1-((Xva_s@w-yva)**2).mean()/yva.var(),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def prep(df):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(np.float64)
    return X

Xtr, Xva = prep(tr), prep(va)
allnan = Xtr.columns[Xtr.isna().all()].tolist()
const = [c for c in Xtr.columns if Xtr[c].nunique(dropna=True)<=1]
print('all-nan cols:', allnan)
print('constant cols:', const[:20], '... total', len(const))
keep = [c for c in cols if c not in allnan and c not in const]
print('kept', len(keep), 'of', len(cols))
Xtr, Xva = Xtr[keep].values, Xva[keep].values
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0)
med = np.where(np.isnan(med), 0, med)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
Xtr = np.nan_to_num(Xtr); Xva = np.nan_to_num(Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
Xtr_s, Xva_s = (Xtr-mu)/sg, (Xva-mu)/sg

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
print('INNER (431):')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s[~iv], ytr[~iv], lam)
    print(' lam', lam, 'inner MAE', round(np.abs(Xtr_s[iv]@w-ytr[iv]).mean(),2))
print('FULL->VAL:')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    p = Xva_s@w
    print(' lam', lam, 'val MAE', round(np.abs(p-yva).mean(),2), 'R2', round(1-((p-yva)**2).mean()/yva.var(),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = m[cols].select_dtypes(include=[np.number])
inf_cols = []
for c in num.columns:
    s = num[c]
    if np.isinf(s.values).any():
        inf_cols.append(c)
print('cols containing inf:', inf_cols)
for c in inf_cols:
    s = num[c].replace([np.inf,-np.inf], np.nan)
    print(c, 'max finite', s.max(), 'n_inf', np.isinf(num[c].values).sum())
print('\nindex col:', m['index'].describe().round(1).to_string())
big = []
for c in num.columns:
    s = num[c].replace([np.inf,-np.inf], np.nan).abs()
    mx = s.max()
    if mx > 1e6: big.append((c, mx))
print('\ncols with |max|>1e6:', big[:20])


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
print('t dtypes sample:', t.dtypes.value_counts().to_dict())
print('t index:', t.index.name, t.index[:3].tolist())
print('tt dtypes:', tt.dtypes.to_dict())
print('tt head:\n', tt.head(3))
print('dup keys in t:', t.duplicated(['household_key','snapshot_day']).sum())
print('dup keys in tt:', tt.duplicated(['household_key','snapshot_day']).sum())
m = t.merge(tt, on=['household_key','snapshot_day'])
print('len t', len(t), 'len m', len(m))
# per-snapshot counts
print(t.groupby('snapshot_day').size().to_string())
print(m.groupby('snapshot_day')['future_spend_4w'].count().to_string())
print('y nan in m:', m['future_spend_4w'].isna().sum())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def prep(df):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(np.float64)

Xtr, Xva = prep(tr), prep(va)
# columns all-NaN in train but with values in validation -> the blowup source
trnan = Xtr.columns[Xtr.isna().all()].tolist()
vanan = Xva.columns[Xva.isna().all()].tolist()
print('all-NaN in TRAIN:', trnan)
print('all-NaN in VAL  :', vanan)
print('NaN in val but not train:', [c for c in cols if Xtr[c].notna().all()==False and c not in trnan][:10])

keep = [c for c in cols if c not in trnan]
Xtr, Xva = Xtr[keep].values, Xva[keep].values
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
Xtr = np.nan_to_num(Xtr); Xva = np.nan_to_num(Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)
dead = sg < 1e-12
print('zero-variance-after-fill cols:', keep.count if False else np.array(keep)[dead])
Xtr_s = np.clip((Xtr-mu)/np.where(dead,1,sg), -10, 10)[:, ~dead]
Xva_s = np.clip((Xva-mu)/np.where(dead,1,sg), -10, 10)[:, ~dead]
print('final feature count', Xtr_s.shape[1])

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
print('INNER(431):', end=' ')
for lam in [1,10,100,1000]:
    w = rfit(Xtr_s[~iv], ytr[~iv], lam)
    print(f'lam{lam}:{np.abs(Xtr_s[iv]@w-ytr[iv]).mean():.2f}', end='  ')
print()
print('FULL->VAL :', end=' ')
for lam in [1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    p = Xva_s@w
    print(f'lam{lam}:{np.abs(p-yva).mean():.2f}', end='  ')
print()


# ---- cell ----

import agent_api, pandas as pd, numpy as np

sd = agent_api.snapshot_days()

def prep_table(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def ridge_svd(X, y, lam):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    b = U.T @ y
    w = Vt.T @ (b * s / (s**2 + lam))
    return w

def evaluate(t, name, drop_extra=()):
    tt = agent_api.train_targets()
    m = t.merge(tt, on=['household_key','snapshot_day'])
    tr = m[m.snapshot_day.isin(sd['train'])]
    cols, prep = prep_table(t, drop_extra)
    Xtr_df = prep(tr)
    keep = Xtr_df.columns[Xtr_df.notna().any()].tolist()
    Xtr = Xtr_df[keep].values
    ytr = tr['future_spend_4w'].values
    med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    mu, sg = Xtr.mean(0), Xtr.std(0)
    dead = sg < 1e-12
    Xtr_s = np.clip((Xtr-mu)/np.where(dead,1,sg), -8, 8)[:, ~dead]
    iv = (tr.snapshot_day==431).values
    out = {}
    for lam in [10, 100, 1000, 10000]:
        w = ridge_svd(Xtr_s[~iv], ytr[~iv], lam)
        inner = np.abs(Xtr_s[iv]@w - ytr[iv]).mean()
        w2 = ridge_svd(Xtr_s, ytr, lam)
        out[lam] = (round(inner,2), round(np.abs(Xtr_s@w2 - ytr).mean(),2))
    print(f'{name}: nfeat={Xtr_s.shape[1]}  (lam: inner431, fulltrain-trainMAE) = {out}')
    return out

# E000 baseline for calibration
b = agent_api.baseline_features()
evaluate(b, 'E000-baseline')
# E009
t9 = agent_api.load_saved('e009_ewma_longlags.parquet')
evaluate(t9, 'E009', drop_extra=('index',))


# ---- cell ----

import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()
T0 = datetime.datetime.now()

def prep_matrix(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=200, lr=0.1, max_depth=4, min_leaf=150, feat_frac=0.4, seed=0, nb=64):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape
    nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1), dtype=np.float64)
    for j in range(p):
        thr[j] = np.unique(np.quantile(Xtr[:,j], qs))
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res_tr = ytr - pred_tr
        tree = {}
        def build(idx, depth, nid):
            g = res_tr[idx]
            tree[nid] = {'leaf': g.mean()}
            if depth>=max_depth or len(idx)<2*min_leaf: return
            best = None
            feats = rng.choice(p, max(1,int(p*feat_frac)), replace=False)
            gc = g.sum(); nn=len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in feats:
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                sl = ss; sr = gc - ss
                var = (sl**2/np.maximum(nl,1)) + (sr**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12: return
            _, j, k = best
            m = Xb[idx,j] <= k
            tree[nid] = {'feat': j, 'thr': thr[j][k], 'left': 2*nid+1, 'right': 2*nid+2}
            build(idx[m], depth+1, 2*nid+1); build(idx[~m], depth+1, 2*nid+2)
        build(np.arange(n), 0, 0)
        # vectorized traversal
        def predict(Xbin):
            n_ = len(Xbin)
            out = np.empty(n_)
            nid = np.zeros(n_, dtype=np.int64)
            active = np.arange(n_)
            while len(active):
                cur = nid[active]
                # find leaf / split
                leaf = np.zeros(len(active), dtype=bool)
                f = np.empty(len(active), dtype=np.int64); th = np.empty(len(active))
                for i, n_ in enumerate(cur):
                    nd = tree[int(n_)]
                    if 'feat' not in nd:
                        leaf[i]=True; out[active[i]] = nd['leaf']
                    else:
                        f[i]=nd['feat']; th[i]=nd['thr']
                if leaf.all(): break
                act = active[~leaf]; f=f[~leaf]; th=th[~leaf]
                # compare Xbin bin index vs bin index of thr
                bidx = np.array([np.searchsorted(thr[ff], tt_) for ff, tt_ in zip(f, th)])
                go_l = Xbin[act, f] <= bidx
                nid[act] = np.where(go_l, np.array([tree[int(n)]['left'] for n in nid[act]]), np.array([tree[int(n)]['right'] for n in nid[act]]))
                active = act
            return out
        dva = predict(Xvb); dtr = predict(Xb)
        pred_va += lr*dva; pred_tr += lr*dtr
    return pred_va, pred_tr

b = agent_api.baseline_features()
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols, prep = prep_matrix(b)
Xtr = prep(tr).values; Xva = prep(va).values
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
med = np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
t0 = datetime.datetime.now()
pv,_ = fit_gbm(Xtr, ytr, Xva, n_trees=120)
print('E000 proxy GBM val MAE', round(np.abs(pv-yva).mean(),2), 'time', round((datetime.datetime.now()-t0).total_seconds(),1), '(harness: 92.4)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()

def prep_matrix(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=200, lr=0.1, max_depth=4, min_leaf=150, feat_frac=0.4, seed=0, nb=64):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape
    nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1), dtype=np.float64)
    for j in range(p):
        u = np.unique(np.quantile(Xtr[:,j], qs))
        thr[j,:len(u)] = u
        if len(u) < nb-1: thr[j,len(u):] = u[-1] if len(u) else 0.0
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res_tr = ytr - pred_tr
        tree = {}
        def build(idx, depth, nid):
            g = res_tr[idx]
            tree[nid] = {'leaf': g.mean()}
            if depth>=max_depth or len(idx)<2*min_leaf: return
            best = None
            feats = rng.choice(p, max(1,int(p*feat_frac)), replace=False)
            gc = g.sum(); nn=len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in feats:
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                sl = ss; sr = gc - ss
                var = (sl**2/np.maximum(nl,1)) + (sr**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12: return
            _, j, k = best
            m = Xb[idx,j] <= k
            tree[nid] = {'feat': j, 'thr': thr[j][k], 'left': 2*nid+1, 'right': 2*nid+2}
            build(idx[m], depth+1, 2*nid+1); build(idx[~m], depth+1, 2*nid+2)
        build(np.arange(n), 0, 0)
        def predict(Xbin):
            n_ = len(Xbin)
            out = np.zeros(n_)
            nid = np.zeros(n_, dtype=np.int64)
            while True:
                cur = nid
                f = np.empty(n_, dtype=np.int64); f.fill(-1)
                thv = np.empty(n_)
                for i in range(n_):
                    nd = tree[int(cur[i])]
                    if 'feat' in nd:
                        f[i] = nd['feat']; thv[i] = nd['thr']
                leaf = f < 0
                out[leaf] += np.array([tree[int(cur[i])]['leaf'] for i in np.where(leaf)[0]])
                if leaf.all(): break
                act = np.where(~leaf)[0]
                bidx = np.array([np.searchsorted(thr[f[i]], thv[i]) for i in act])
                go_l = Xbin[act, f[act]] <= bidx
                nid[act] = np.where(go_l, [tree[int(n)]['left'] for n in nid[act]], [tree[int(n)]['right'] for n in nid[act]])
            return out
        dva = predict(Xvb); dtr = predict(Xb)
        pred_va += lr*dva; pred_tr += lr*dtr
    return pred_va, pred_tr

b = agent_api.baseline_features()
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols, prep = prep_matrix(b)
Xtr = prep(tr).values; Xva = prep(va).values
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
med = np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
t0 = datetime.datetime.now()
pv,_ = fit_gbm(Xtr, ytr, Xva, n_trees=120)
print('E000 proxy GBM val MAE', round(np.abs(pv-yva).mean(),2), 'time', round((datetime.datetime.now()-t0).total_seconds(),1), '(harness: 92.4)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()

def prep_matrix(t, drop_extra=()):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=200, lr=0.1, max_depth=4, min_leaf=150, feat_frac=0.4, seed=0, nb=64):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape
    nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1), dtype=np.float64)
    for j in range(p):
        u = np.unique(np.quantile(Xtr[:,j], qs))
        thr[j,:len(u)] = u
        if len(u): thr[j,len(u):] = u[-1]
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res_tr = ytr - pred_tr
        tree = {}
        def build(idx, depth, nid):
            g = res_tr[idx]
            tree[nid] = {'leaf': g.mean()}
            if depth>=max_depth or len(idx)<2*min_leaf: return
            best = None
            feats = rng.choice(p, max(1,int(p*feat_frac)), replace=False)
            gc = g.sum(); nn=len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in feats:
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                sl = ss; sr = gc - ss
                var = (sl**2/np.maximum(nl,1)) + (sr**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12: return
            _, j, k = best
            m = Xb[idx,j] <= k
            tree[nid] = {'feat': j, 'thr': thr[j][k], 'left': 2*nid+1, 'right': 2*nid+2}
            build(idx[m], depth+1, 2*nid+1); build(idx[~m], depth+1, 2*nid+2)
        build(np.arange(n), 0, 0)
        def predict(Xbin):
            n_ = len(Xbin)
            out = np.zeros(n_)
            nid = np.zeros(n_, dtype=np.int64)
            active = np.arange(n_)
            while len(active):
                cur = nid[active]
                has = np.array(['feat' in tree[int(c)] for c in cur])
                li = np.where(~has)[0]
                if len(li):
                    out[active[li]] += [tree[int(cur[i])]['leaf'] for i in li]
                act = active[has]
                if len(act)==0: break
                cur = nid[act]
                f = np.array([tree[int(c)]['feat'] for c in cur])
                th = np.array([tree[int(c)]['thr'] for c in cur])
                bidx = np.array([np.searchsorted(thr[ff], tt_) for ff, tt_ in zip(f, th)])
                go_l = Xbin[act, f] <= bidx
                nid[act] = np.where(go_l, [tree[int(c)]['left'] for c in cur], [tree[int(c)]['right'] for c in cur])
                active = act
            return out
        dva = predict(Xvb); dtr = predict(Xb)
        pred_va += lr*dva; pred_tr += lr*dtr
    return pred_va, pred_tr

b = agent_api.baseline_features()
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols, prep = prep_matrix(b)
Xtr = prep(tr).values; Xva = prep(va).values
ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
med = np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
t0 = datetime.datetime.now()
pv,_ = fit_gbm(Xtr, ytr, Xva, n_trees=120)
print('E000 proxy GBM val MAE', round(np.abs(pv-yva).mean(),2), 'time', round((datetime.datetime.now()-t0).total_seconds(),1), '(harness: 92.4)')


# ---- cell ----

import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()
b = agent_api.baseline_features()
print('baseline cols:', b.columns.tolist())
print('baseline dtypes:', b.dtypes.to_dict())
tt = agent_api.train_targets()
m = b.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
for c in b.columns:
    if c in ('household_key','snapshot_day'): continue
    print(c, 'train_nan%', round(tr[c].isna().mean(),3), 'val_nan%', round(va[c].isna().mean(),3), 'dtype', b[c].dtype)


# ---- cell ----

import agent_api, pandas as pd, numpy as np, datetime

sd = agent_api.snapshot_days()

def prep_matrix(t, drop_extra=('index',)):
    cols = [c for c in t.columns if c not in ('household_key','snapshot_day') and c not in drop_extra]
    def prep(df):
        X = df[cols].copy()
        for c in cols:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
        return X.astype(np.float64)
    return cols, prep

def fit_gbm(Xtr, ytr, Xva, n_trees=250, lr=0.08, max_depth=4, min_leaf=200, feat_frac=0.35, seed=0, nb=48):
    rng = np.random.RandomState(seed)
    n, p = Xtr.shape; nv = Xva.shape[0]
    qs = np.linspace(0,1,nb+2)[1:-1]
    thr = np.zeros((p, nb-1))
    for j in range(p):
        u = np.unique(np.quantile(Xtr[:,j], qs))
        thr[j,:len(u)] = u
        if len(u): thr[j,len(u):] = u[-1]
    Xb = np.empty((n,p), dtype=np.int32); Xvb = np.empty((nv,p), dtype=np.int32)
    for j in range(p):
        Xb[:,j] = np.searchsorted(thr[j], Xtr[:,j])
        Xvb[:,j] = np.searchsorted(thr[j], Xva[:,j])
    pred_va = np.full(nv, ytr.mean()); pred_tr = np.full(n, ytr.mean())
    for t_i in range(n_trees):
        res = ytr - pred_tr
        nodes = {}; queue = [(np.arange(n), 0, 0)]
        order = []
        while queue:
            idx, depth, nid = queue.pop(0)
            g = res[idx]
            if depth >= max_depth or len(idx) < 2*min_leaf:
                nodes[nid] = ('leaf', g.mean()); continue
            best = None
            gc = g.sum(); nn = len(idx)
            base = (g**2).sum() - gc**2/nn
            for j in rng.choice(p, max(1,int(p*feat_frac)), replace=False):
                hj = Xb[idx,j]
                cnt = np.bincount(hj, minlength=nb)
                s = np.bincount(hj, weights=g, minlength=nb)
                c = np.cumsum(cnt)[:-1]; ss = np.cumsum(s)[:-1]
                nl = c; nr = nn - c
                ok = (nl>=min_leaf)&(nr>=min_leaf)
                if not ok.any(): continue
                var = (ss**2/np.maximum(nl,1)) + ((gc-ss)**2/np.maximum(nr,1))
                var[~ok] = -np.inf
                k = int(var.argmax())
                if var[k] > -np.inf and (best is None or var[k] > best[0]):
                    best = (var[k], j, k)
            if best is None or best[0] <= base - 1e-12:
                nodes[nid] = ('leaf', g.mean()); continue
            _, j, k = best
            m = Xb[idx,j] <= k
            nodes[nid] = ('split', j, k, 2*nid+1, 2*nid+2)
            order.append(nid)
            queue.append((idx[m], depth+1, 2*nid+1)); queue.append((idx[~m], depth+1, 2*nid+2))
        def predict(Xbin):
            nid = np.zeros(len(Xbin), dtype=np.int64)
            for nid_ in order:
                _, j, k, l, r = nodes[nid_]
                msk = nid == nid_
                go_l = Xbin[msk, j] <= k
                nid[msk] = np.where(go_l, l, r)
            return np.array([nodes[i][1] for i in nid])
        pred_va += lr*predict(Xvb); pred_tr += lr*predict(Xb)
    return pred_va

def run(name, t):
    tt = agent_api.train_targets()
    m = t.merge(tt, on=['household_key','snapshot_day'])
    tr_fit = m[m.snapshot_day.isin(sd['train'][:-1])]   # 95..403
    tr_ev  = m[m.snapshot_day==431]
    cols, prep = prep_matrix(t)
    Xf = prep(tr_fit); keep = Xf.columns[Xf.notna().any()].tolist()
    Xf = Xf[keep].values; Xe = prep(tr_ev)[keep].values
    yf = tr_fit['future_spend_4w'].values; ye = tr_ev['future_spend_4w'].values
    med = np.nanmedian(Xf,0); med = np.where(np.isnan(med),0,med)
    Xf = np.where(np.isnan(Xf), med, Xf); Xe = np.where(np.isnan(Xe), med, Xe)
    t0 = datetime.datetime.now()
    pv = fit_gbm(Xf, yf, Xe)
    print(f'{name}: inner431 MAE {np.abs(pv-ye).mean():.2f}  nfeat {len(keep)}  ({(datetime.datetime.now()-t0).total_seconds():.0f}s)')

run('E000', agent_api.baseline_features())
run('E001', agent_api.load_saved('recency_agg.parquet'))
run('E005', agent_api.load_saved('basket_tenure.parquet'))
run('E009', agent_api.load_saved('e009_ewma_longlags.parquet'))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t9 = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t9.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]

# --- diagnostics ---
print('corr(index, y) train:', round(np.corrcoef(tr['index'], tr['future_spend_4w'])[0,1], 4))
g = tr[tr.snapshot_day <= 403].groupby('household_key')['future_spend_4w'].mean()
ev = tr[tr.snapshot_day == 431]
pred = ev['household_key'].map(g).fillna(tr['future_spend_4w'].mean())
print('oracle household-mean MAE @431:', round(np.abs(pred - ev['future_spend_4w']).mean(), 2),
      '| frac hh w/o earlier snap:', round(ev['household_key'].isin(g.index).mean(), 3))

# --- prune list (train-determined) ---
feat_cols = [c for c in t9.columns if c not in ('household_key','snapshot_day')]
const_cols = [c for c in feat_cols if tr[c].nunique(dropna=True) <= 1]
allnan_cols = [c for c in feat_cols if tr[c].isna().all()]
drop = sorted(set(['index'] + const_cols + allnan_cols))
print('dropping', len(drop), 'cols; sample:', drop[:6])
e011 = t9.drop(columns=drop)
print('e011 shape', e011.shape)

# --- robust-distribution features (as-of safe) ---
def make_feats(view, snapshot_day):
    s = snapshot_day
    hh = pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.day > s - 364]
    out = pd.DataFrame(index=hh)
    z = np.zeros((len(hh), 26))
    T = np.zeros((len(hh), 13))
    if len(tx):
        h = tx.household_key.values; d = tx.day.values; v = tx.sales_value.values
        tmp = pd.DataFrame({'h': h, 'w': (s - d) // 7, 'v': v})
        piv = tmp.pivot_table(index='h', columns='w', values='v', aggfunc='sum').reindex(hh).reindex(columns=range(26)).fillna(0.0)
        W = piv.values
        tmp2 = pd.DataFrame({'h': h, 'k': (s - d) // 28, 'v': v})
        piv2 = tmp2.pivot_table(index='h', columns='k', values='v', aggfunc='sum').reindex(hh).reindex(columns=range(13)).fillna(0.0)
        T = piv2.values
        r28 = tx[tx.day > s - 28]
        out['rb_max28'] = r28.groupby('household_key')['sales_value'].max().reindex(hh).fillna(0.0).values
        r84 = tx[tx.day > s - 84]
        nb = r84.groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0).values
        s84 = r84.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
        out['rb_max28_ratio'] = out['rb_max28'] / (s84 / np.maximum(nb, 1) + 1.0)
        out['rb_spend3'] = tx[tx.day > s - 3].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0).values
        out['rb_trips7'] = tx[tx.day > s - 7].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0).values
    out['rb_wmed26'] = np.median(W, axis=1)
    out['rb_wq25'] = np.percentile(W, 25, axis=1)
    out['rb_wq75'] = np.percentile(W, 75, axis=1)
    out['rb_wiqr'] = out['rb_wq75'] - out['rb_wq25']
    zz = (W == 0)
    out['rb_wzstreak'] = np.cumprod(zz, axis=1).sum(axis=1)
    out['rb_wnz4'] = (~zz[:, :4]).sum(axis=1)
    out['rb_tmed13'] = np.median(T, axis=1)
    out['rb_tq25'] = np.percentile(T, 25, axis=1)
    out['rb_tq75'] = np.percentile(T, 75, axis=1)
    out['rb_tmin13'] = T.min(axis=1)
    out['rb_tzero13'] = (T == 0).sum(axis=1)
    out['rb_tcv13'] = T.std(axis=1) / (T.mean(axis=1) + 1.0)
    return out

feats = agent_api.build_features(make_feats)
fcols = [c for c in feats.columns if c.startswith('rb_')]
print('rb feats:', len(fcols), 'nan%:', round(float(feats[fcols].isna().mean().mean()), 4))
print(feats[fcols].describe().loc[['mean','50%']].round(1).to_string())

feats['household_key'] = feats['household_key'].astype(e011.household_key.dtype)
feats['snapshot_day'] = feats['snapshot_day'].astype(e011.snapshot_day.dtype)
e012 = e011.merge(feats[['household_key','snapshot_day'] + fcols], on=['household_key','snapshot_day'], how='left')
print('e012 shape', e012.shape, 'rb nan in e012:', int(e012[fcols].isna().sum().sum()))
p1 = agent_api.save_table(e011, 'e011_pruned.parquet')
p2 = agent_api.save_table(e012, 'e012_robust.parquet')
print('saved:', p1, '|', p2)
