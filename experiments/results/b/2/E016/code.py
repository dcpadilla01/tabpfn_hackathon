
import agent_api as A

for name in ['e014_base','e014_gbm_oob','e014_stack','e013_denoise']:
    try:
        t = A.load_saved(name + '.parquet')
        print('===', name, t.shape)
        print(t.columns.tolist()[:40])
        print('snapshot_days:', sorted(t.snapshot_day.unique()))
        print(t.head(3))
        print()
    except Exception as e:
        print(name, 'ERR', e)


# ---- cell ----

import agent_api as A
for name in ['e014_base','e014_gbm_oob','e014_stack']:
    t = A.load_saved(name + '.parquet')
    print('===', name, t.shape)
    print(t.columns.tolist()[-30:])


# ---- cell ----

import agent_api as A
t = A.load_saved('e014_gbm_oob.parquet')
print(t['gbm_corr'].describe())
print('nan frac', t['gbm_corr'].isna().mean())
print('per-snapshot corr mean:')
print(t.groupby('snapshot_day')['gbm_corr'].mean())


# ---- cell ----

import agent_api as A
t = A.load_saved('e014_stack.parquet')
print(t['gbm_pred'].describe())
print(t.groupby('snapshot_day')['gbm_pred'].mean())
print(t.groupby('snapshot_day').size())


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd

e13 = A.load_saved('e013_denoise.parquet')
st  = A.load_saved('e014_stack.parquet')[['household_key','snapshot_day','gbm_pred']]
tt  = A.train_targets()
df = e13.merge(st, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day.isin(A.snapshot_days()['train'])].copy()
print('train rows', tr.shape)

def design(frame, cols):
    X = frame[cols].copy()
    num = X.select_dtypes(include=[np.number]).astype(float)
    cat = X.select_dtypes(exclude=[np.number])
    parts = [num]
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat.astype(str), dummy_na=True).astype(float))
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf,-np.inf], np.nan).fillna(X.median())

def ridge_cv(X, y, alpha=100.0, folds=5, seed=0):
    Xv = X.values.astype(float)
    mu = np.nanmean(Xv,0); sd = Xv.std(0); sd[sd==0]=1
    Xv = (Xv-mu)/sd
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    fs = np.array_split(idx, folds)
    maes=[]
    for f in range(folds):
        b = fs[f]; a = np.concatenate([fs[j] for j in range(folds) if j!=f])
        Xa = np.hstack([Xv[a], np.ones((len(a),1))])
        Xb = np.hstack([Xv[b], np.ones((len(b),1))])
        A_ = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); A_[-1,-1] -= alpha
        w = np.linalg.solve(A_, Xa.T@y[a])
        maes.append(np.abs(Xb@w - y[b]).mean())
    return np.mean(maes), np.std(maes)

num = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
y = tr.future_spend_4w.values
for cols, name in [(num,'E013'), (num+['gbm_pred'],'E013+gbm_pred')]:
    X = design(tr, cols)
    m,s = ridge_cv(X, y)
    print(f'{name}: {m:.3f} +/- {s:.3f}')
tr['gbm_sq'] = tr.gbm_pred**2
X = design(tr, num+['gbm_pred','gbm_sq']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq: {m:.3f} +/- {s:.3f}')
X = design(tr, num+['gbm_pred','gbm_sq','gbm_cu']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq+cu: {m:.3f} +/- {s:.3f}') if False else None
tr['gbm_cu'] = tr.gbm_pred**3
X = design(tr, num+['gbm_pred','gbm_sq','gbm_cu']); m,s = ridge_cv(X,y); print(f'E013+gbm+sq+cu: {m:.3f} +/- {s:.3f}')


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd

e13 = A.load_saved('e013_denoise.parquet')
st  = A.load_saved('e014_stack.parquet')[['household_key','snapshot_day','gbm_pred']]
tt  = A.train_targets()
df = e13.merge(st, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day.isin(A.snapshot_days()['train'])].copy()
y = tr.future_spend_4w.values

def design(frame, cols):
    X = frame[cols].copy()
    num = X.select_dtypes(include=[np.number]).astype(float)
    cat = X.select_dtypes(exclude=[np.number])
    parts = [num]
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat.astype(str), dummy_na=True).astype(float))
    X = pd.concat(parts, axis=1)
    return X.replace([np.inf,-np.inf], np.nan).fillna(X.median())

def ridge_cv(X, y, alpha=100.0, folds=5, seed=0):
    Xv = X.values.astype(float)
    mu = np.nanmean(Xv,0); sd = Xv.std(0); sd[sd==0]=1
    Xv = (Xv-mu)/sd
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y)); fs = np.array_split(idx, folds)
    maes=[]
    for f in range(folds):
        b = fs[f]; a = np.concatenate([fs[j] for j in range(folds) if j!=f])
        Xa = np.hstack([Xv[a], np.ones((len(a),1))]); Xb = np.hstack([Xv[b], np.ones((len(b),1))])
        M = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); M[-1,-1] -= alpha
        w = np.linalg.solve(M, Xa.T@y[a])
        maes.append(np.abs(Xb@w - y[b]).mean())
    return np.mean(maes)

num = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
tr['gbm_log'] = np.log1p(tr.gbm_pred.clip(lower=0))
tr['gbm_sqrt'] = np.sqrt(tr.gbm_pred.clip(lower=0))
for cols, name in [
    (['gbm_pred'], 'gbm_pred alone'),
    (num+['gbm_log'], 'E013+gbm_log'),
    (num+['gbm_sqrt'], 'E013+gbm_sqrt'),
    (num+['gbm_pred','gbm_log'], 'E013+gbm_pred+gbm_log'),
]:
    print(f'{name}: {ridge_cv(design(tr, cols), y):.3f}')


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd

e13 = A.load_saved('e013_denoise.parquet')
st  = A.load_saved('e014_stack.parquet')[['household_key','snapshot_day','gbm_pred']]
tt  = A.train_targets()
df = e13.merge(st, on=['household_key','snapshot_day'], how='left').merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = df[df.snapshot_day.isin(A.snapshot_days()['train'])].copy()
y = tr.future_spend_4w.values

def design(frame, cols):
    X = frame[cols].copy()
    num = X.select_dtypes(include=[np.number]).astype(float)
    cat = X.select_dtypes(exclude=[np.number])
    parts = [num]
    if cat.shape[1]:
        parts.append(pd.get_dummies(cat.astype(str), dummy_na=True).astype(float))
    return pd.concat(parts, axis=1).replace([np.inf,-np.inf], np.nan).fillna(X.median())

def ridge_cv(X, y, alpha=100.0, folds=5, seed=0):
    Xv = X.values.astype(float)
    mu = np.nanmean(Xv,0); sd = Xv.std(0); sd[sd==0]=1
    Xv = (Xv-mu)/sd
    rng = np.random.RandomState(seed); idx = rng.permutation(len(y)); fs = np.array_split(idx, folds)
    maes=[]
    for f in range(folds):
        b = fs[f]; a = np.concatenate([fs[j] for j in range(folds) if j!=f])
        Xa = np.hstack([Xv[a], np.ones((len(a),1))]); Xb = np.hstack([Xv[b], np.ones((len(b),1))])
        M = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); M[-1,-1] -= alpha
        w = np.linalg.solve(M, Xa.T@y[a])
        maes.append(np.abs(Xb@w - y[b]).mean())
    return np.mean(maes)

num = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
tr['gbm_log'] = np.log1p(tr.gbm_pred.clip(lower=0))
X = design(tr, num+['gbm_pred','gbm_log'])
for al in [30., 100., 300., 1000.]:
    print(f'alpha={al}: {ridge_cv(X, y, alpha=al):.3f}')
# also check per-fold val snapshots behave (train-only CV by snapshot blocks)
print('done')
