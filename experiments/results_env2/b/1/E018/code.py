t8 = agent_api.load_saved('e008_level_shape.parquet')
print('E008', t8.shape)
print(sorted([c for c in t8.columns]))
t16 = agent_api.load_saved('e016_grid.parquet')
print('E016', t16.shape)
print(sorted([c for c in t16.columns])[:60])
print(agent_api.snapshot_days())


# ---- cell ----
import numpy as np, pandas as pd
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
print('t16', t16.shape, 't8', t8.shape, 'tt', tt.shape)
gcols = [c for c in t16.columns if c.startswith('g_')]
print('grid cols:', gcols)
print('t8 cols subset of t16:', set(t8.columns) <= set(t16.columns))
df = t16.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
y = df['future_spend_4w'].values
print('target mean %.2f std %.2f zero-share %.3f median %.2f' % (y.mean(), y.std(), (y==0).mean(), np.median(y)))
tr_days = agent_api.snapshot_days()['train']
print('train days', tr_days)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t17 = agent_api.load_saved('e017_grand.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
print('t17', t17.shape, 't16', t16.shape)
pool = t17.merge(t16, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
print('pool', pool.shape)
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('df', df.shape)

# column types
num_cols, cat_cols = [], []
for c in df.columns:
    if c in ('household_key','snapshot_day','future_spend_4w'): continue
    if df[c].dtype.kind in 'ifb': num_cols.append(c)
    else: cat_cols.append(c)
print('num', len(num_cols), 'cat', cat_cols)
for c in cat_cols: print(c, df[c].nunique(), df[c].dtype)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t9  = agent_api.load_saved('e009_target_enc.parquet')
t16 = agent_api.load_saved('e016_grid.parquet')
tt  = agent_api.train_targets()
df9 = t9.merge(tt, on=['household_key','snapshot_day'])
df16 = t16.merge(tt, on=['household_key','snapshot_day'])
y9 = df9['future_spend_4w'].values; y16 = df16['future_spend_4w'].values

te_cols = [c for c in t9.columns if c not in t8_cols] if False else None
t8 = agent_api.load_saved('e008_level_shape.parquet')
te_cols = [c for c in t9.columns if c not in set(t8.columns)]
print('E009 added cols:', te_cols)
for c in te_cols:
    x = df9[c].astype(float).values
    ok = ~np.isnan(x)
    print('%-14s corr=%+.3f  nan%%=%.1f  uniq=%d' % (c, np.corrcoef(x[ok], y9[ok])[0,1], 100*(~ok).mean(), len(np.unique(x[ok]))))

gcols = [c for c in t16.columns if c.startswith('g_')]
print('\ngrid feature corr with target (train rows):')
rows=[]
for c in gcols:
    x = df16[c].astype(float).values; ok=~np.isnan(x)
    rows.append((c, np.corrcoef(x[ok], y16[ok])[0,1], 100*(~ok).mean()))
for c,r,nn in sorted(rows, key=lambda t:-abs(t[1]))[:15]:
    print('%-12s corr=%+.3f nan%%=%.0f' % (c,r,nn))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]

num_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and df[c].dtype.kind in 'ifb']
# drop exact-duplicate columns (from the _d suffix merge)
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
print('num cols after dedup:', len(num_cols))

itr = df[df.snapshot_day.isin(itr_days)]; iva = df[df.snapshot_day.isin(iva_days)]
Xtr = itr[num_cols].astype(float).values; Xva = iva[num_cols].astype(float).values
ytr = itr['future_spend_4w'].values;     yva = iva['future_spend_4w'].values
mu, sd = np.nanmean(Xtr,0), np.nanstd(Xtr,0); sd[sd==0]=1
Ztr = np.where(np.isnan(Xtr),0,(Xtr-mu)/sd); Zva = np.where(np.isnan(Xva),0,(Xva-mu)/sd)
Ztr = np.c_[Ztr, np.ones(len(Ztr))]; Zva = np.c_[Zva, np.ones(len(Zva))]

def ridge_fit(Z,y,lam):
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Z.T@y)

# log-target ridge usually better for skewed spend; predict expm1 with smearing
ly = np.log1p(ytr)
best=None
for lam in [10,30,100,300,1000]:
    w = ridge_fit(Ztr, ly, lam)
    p = np.expm1(Zva@w); s = np.mean(np.exp(Ztr@w - (Ztr@w)))  # smearing ~1 on train
    m = np.mean(np.abs(p-yva))
    pw = np.expm1(Ztr@w); mw = np.mean(np.abs(pw-ytr))
    print('lam=%5d  inner-val MAE %.3f (train MAE %.3f)' % (lam, m, mw))
    if best is None or m<best[1]: best=(lam,m,w)
print('best lam', best[0])


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and df[c].dtype.kind in 'ifb']
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep

itr = df[df.snapshot_day.isin(itr_days)]; iva = df[df.snapshot_day.isin(iva_days)]
def prep(d):
    X = d[num_cols].astype(float).values
    return X, d['future_spend_4w'].values
Xtr,ytr = prep(itr); Xva,yva = prep(iva)
mu, sd = np.nanmean(Xtr,0), np.nanstd(Xtr,0); sd[sd==0]=1
Ztr = np.where(np.isnan(Xtr),0,(Xtr-mu)/sd); Zva = np.where(np.isnan(Xva),0,(Xva-mu)/sd)
Ztr = np.c_[Ztr, np.ones(len(Ztr))]; Zva = np.c_[Zva, np.ones(len(Zva))]
CAP = np.log1p(2000.0)
def ridge(Z,y,lam):
    A = Z.T@Z + lam*np.eye(Z.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Z.T@y)

print('baseline mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
for lam in [10,30,100,300,1000,3000]:
    w = ridge(Ztr, np.log1p(ytr), lam)
    f = lambda Z: np.expm1(np.clip(Z@w, -5, CAP))
    smear = np.mean(ytr/np.maximum(f(Ztr),1e-9))
    print('lam=%5d log-target: inner MAE %.3f | raw-target MAE %.3f' % (lam, np.mean(np.abs(f(Zva)*smear-yva)), np.mean(np.abs(f(Ztr)*smear-ytr))))
# raw target
for lam in [100,1000,3000,10000]:
    w = ridge(Ztr, ytr, lam)
    print('lam=%5d raw-target : inner MAE %.3f | train MAE %.3f' % (lam, np.mean(np.abs(Zva@w-yva)), np.mean(np.abs(Ztr@w-ytr))))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and df[c].dtype.kind in 'ifb']
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
X = df[num_cols].astype(float)
bad = [c for c in num_cols if np.isinf(X[c].values).any()]
print('cols with inf:', bad)
for c in bad: print(c, df[c].describe())

Xv = X.values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days); iva = df.snapshot_day.isin(iva_days)
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr.values],0); sd = np.nanstd(Xv[itr.values],0); sd[sd==0]=1
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd)
Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr.values], Z[iva.values]
ytr, yva = y[itr.values], y[iva.values]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
print('mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
best=(None,1e9)
for lam in [30,100,300,1000,3000,10000]:
    w = ridge(Ztr, np.log1p(ytr), lam)
    f = lambda Z: np.expm1(np.clip(Z@w, -5, CAP))
    smear = np.mean(ytr/np.maximum(f(Ztr),1e-9))
    m = np.mean(np.abs(f(Zva)*smear-yva)); mt = np.mean(np.abs(f(Ztr)*smear-ytr))
    w2 = ridge(Ztr, ytr, lam)
    m2 = np.mean(np.abs(Zva@w2-yva))
    print('lam=%5d log: %.3f (tr %.3f) | raw: %.3f (tr %.3f)' % (lam, m, mt, m2, np.mean(np.abs(Ztr@w2-ytr))))
    best = min(best, [(lam,'log',m),(lam,'raw',m2)][np.argmin([m,m2])][::2]+[0]) if False else best
    if m<best[1]: best=(('log',lam),m)
    if m2<best[1]: best=(('raw',lam),m2)
print('best inner:', best)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day',], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and df[c].dtype.kind in 'ifb']
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days); iva = df.snapshot_day.isin(iva_days)
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr.values],0); sd = np.nanstd(Xv[itr.values],0); sd[sd==0]=1
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd); Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr.values], Z[iva.values]
ytr, yva = y[itr.values], y[iva.values]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
w = ridge(Ztr, np.log1p(ytr), 1000)
pred_log = np.expm1(np.clip(Zva@w, -5, CAP))
print('nan in Zva@w:', np.isnan(Zva@w).sum(), ' max log pred:', np.nanmax(Zva@w))
print('nan in pred_log:', np.isnan(pred_log).sum())
ly = Zva@w
print('finite?', np.isfinite(ly).sum(), '/', len(ly))
smear = np.mean(ytr/np.maximum(np.expm1(np.clip(Ztr@w,-5,CAP)),1e-9))
print('smear', smear)
m = np.mean(np.abs(pred_log*smear - yva))
print('log-target lam=1000 inner MAE:', m)
# where do nans come from?
if np.isnan(ly).any():
    idx = np.where(np.isnan(ly))[0][:5]
    print('bad rows:', iva.values.nonzero()[0][idx])


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if c not in ('0','x') and df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days); iva = df.snapshot_day.isin(iva_days)
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr.values],0); sd = np.nanstd(Xv[itr.values],0); sd[sd==0]=1
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd); Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr.values], Z[iva.values]
ytr, yva = y[itr.values], y[iva.values]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
print('mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
best=None
for lam in [30,100,300,1000,3000,10000]:
    w = ridge(Ztr, np.log1p(ytr), lam)
    f = lambda Z: np.expm1(np.clip(Z@w, -5, CAP))
    smear = np.mean(ytr/np.maximum(f(Ztr),1e-9))
    m  = np.mean(np.abs(f(Zva)*smear-yva)); mt = np.mean(np.abs(f(Ztr)*smear-ytr))
    w2 = ridge(Ztr, ytr, lam)
    m2 = np.mean(np.abs(Zva@w2-yva));       m2t= np.mean(np.abs(Ztr@w2-ytr))
    print('lam=%5d log: %.3f (tr %.3f) | raw: %.3f (tr %.3f)' % (lam, m, mt, m2, m2t))
    if best is None or min(m,m2)<best[1]:
        best = (('log' if m<m2 else 'raw', lam), min(m,m2))
print('best inner:', best)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days).values; iva = df.snapshot_day.isin(iva_days).values
y = df['future_spend_4w'].values
print('nan y:', np.isnan(y).sum(), 'overlap itr&iva:', (itr&iva).sum(), 'neither:', (~itr&~iva).sum())
mu = np.nanmean(Xv[itr],0); sd = np.nanstd(Xv[itr],0); sd[sd==0]=1
print('nan in mu:', np.isnan(mu).sum())
Z = np.where(np.isnan(Xv),0,(Xv-mu)/sd)
print('nan in Z:', np.isnan(Z).sum())
Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr], Z[iva]
ytr, yva = y[itr], y[iva]
print('nan Zva:', np.isnan(Zva).sum(), 'nan yva:', np.isnan(yva).sum(), 'nan ytr:', np.isnan(ytr).sum())
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
w2 = ridge(Ztr, ytr, 1000)
print('nan w2:', np.isnan(w2).sum())
p = Zva@w2
print('nan p:', np.isnan(p).sum())
m = np.mean(np.abs(p-yva))
print('raw inner MAE:', m)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days).values; iva = df.snapshot_day.isin(iva_days).values
y = df['future_spend_4w'].values
mu = np.nanmean(Xv[itr],0)
bad = [num_cols[i] for i in np.where(np.isnan(mu))[0]]
print('all-NaN-on-inner-train cols:', bad)
for c in bad:
    print(c, 'nan frac overall %.2f' % df[c].isna().mean(), 'by day:', df.groupby('snapshot_day')[c].apply(lambda s: s.isna().mean().round(2)).to_dict())


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
t16 = agent_api.load_saved('e016_grid.parquet')
t8  = agent_api.load_saved('e008_level_shape.parquet')
tt  = agent_api.train_targets()
pool = t16.merge(t8, on=['household_key','snapshot_day'], how='outer', suffixes=('','_d'))
df = pool.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr_days = agent_api.snapshot_days()['train']
iva_days = tr_days[-3:]; itr_days = tr_days[:-3]
num_cols = [c for c in df.columns if df[c].dtype.kind in 'ifb' and c not in ('household_key','snapshot_day','future_spend_4w')]
keep, seen = [], {}
for c in num_cols:
    h = pd.util.hash_pandas_object(df[c].fillna(0), index=False).values.sum()
    if h in seen: continue
    seen[h]=c; keep.append(c)
num_cols = keep
Xv = df[num_cols].astype(float).values.copy()
Xv[~np.isfinite(Xv)] = np.nan
itr = df.snapshot_day.isin(itr_days).values; iva = df.snapshot_day.isin(iva_days).values
y = df['future_spend_4w'].values
# fill: column median on inner-train (all-nan cols -> 0)
med = np.nanmedian(np.where(itr[:,None], Xv, np.nan), axis=0)
med = np.where(np.isnan(med), 0.0, med)
Xf = np.where(np.isnan(Xv), med, Xv)
mu = Xf[itr].mean(0); sd = Xf[itr].std(0); sd[sd==0]=1
Z = (Xf-mu)/sd; Z = np.c_[Z, np.ones(len(Z))]
Ztr, Zva = Z[itr], Z[iva]; ytr, yva = y[itr], y[iva]
def ridge(Zt,yt,lam):
    A = Zt.T@Zt + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    return np.linalg.solve(A, Zt.T@yt)
CAP = np.log1p(2000.0)
print('mean-pred inner MAE %.2f' % np.mean(np.abs(yva-ytr.mean())))
def evalset(cols_idx, tag):
    out=[]
    for lam in [30,100,300,1000,3000]:
        w = ridge(Ztr[:,list(cols_idx)+[-1]], np.log1p(ytr), lam)
        f = lambda Z: np.expm1(np.clip(Z@w,-5,CAP))
        smear = np.mean(ytr/np.maximum(f(Ztr[:,list(cols_idx)+[-1]]),1e-9))
        out.append((lam, np.mean(np.abs(f(Zva[:,list(cols_idx)+[-1]])*smear-yva))))
    lam,m = min(out, key=lambda t:t[1])
    print('%-22s best lam=%5d inner MAE %.3f' % (tag, lam, m))
    return m
allidx = list(range(len(num_cols)))
e017idx = [i for i,c in enumerate(num_cols) if not c.endswith('_d') and c not in [g for g in num_cols if g.startswith('g_')]]
grididx = [i for i,c in enumerate(num_cols) if c.startswith('g_')]
extra   = [i for i,c in enumerate(num_cols) if c.endswith('_d') and c not in [g for g in num_cols if g.startswith('g_')]]
m_all = evalset(allidx, 'ALL %d'%len(allidx))
m_17  = evalset(e017idx, 'E017 %d'%len(e017idx))
m_g   = evalset(grididx, 'grid only %d'%len(grididx))
m_17g = evalset(e017idx+grididx, 'E017+grid %d'%(len(e017idx)+len(grididx)))
m_all2= evalset(e017idx+grididx+extra, 'E017+grid+extra %d'%(len(e017idx)+len(grididx)+len(extra)))


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t13 = agent_api.load_saved('e013_union.parquet')   # E008+E006+E007 union, best table (60.703)
print('E013', t13.shape)

df = t13.copy()
# --- new feature block: non-linear/interaction level features ---
eps = 1e-9
# 1) sqrt (concave) transforms of top heavy-tailed level features -> closer to conditional median
for c in ['sp28','sp84','sp364','g_wmean','z_med4w_hist','sp28_rate','sp364_rate']:
    if c in df.columns:
        df['sq_'+c] = np.sqrt(np.clip(df[c].astype(float), 0, None))
# 2) empirical-Bayes shrunk expected next-4w spend: rate shrunk toward global daily rate
m_day = df['sp364'].astype(float).sum() / (df['sp364'].notna().sum()*364.0)
for w, k in [(84,56),(168,56),(364,112)]:
    sp = df['sp%d'%w].astype(float)
    df['exp_eb%d'%w] = 28.0 * (sp.fillna(0) + k*m_day) / (w + k)
# 3) zero-probability-adjusted expectation
zr = df['z_zero_rate_hist'].astype(float).fillna(0.5).clip(0,1)
df['expz84'] = df['exp_eb84'] * (1-zr)
df['expz364'] = df['exp_eb364'] * (1-zr)
# 4) recency-weighted and cadence-weighted level interactions
dsl = df['days_since_last'].astype(float).fillna(999)
df['i_rec_level']  = df['sp28'].astype(float).fillna(0) * np.exp(-dsl/28.0)
df['i_trips_lvl']  = df['sp28_rate'].astype(float).fillna(0) * df['trips28'].astype(float).fillna(0)
df['i_act_lvl']    = df['sp84_rate'].astype(float).fillna(0) * df['nact84'].astype(float).fillna(0)
df['i_med_gw']     = df['z_med4w_hist'].astype(float).fillna(0) * df['g_wmean'].astype(float).fillna(0)
df['i_ew_gw']      = df['g_ew6'].astype(float).fillna(0) * df['g_wmean'].astype(float).fillna(0)
# 5) blend of the two best level estimators
df['lvl_blend'] = 0.5*df['exp_eb84'].fillna(0) + 0.25*df['exp_eb364'].fillna(0) + 0.25*df['g_wmean'].fillna(0)

newc = [c for c in df.columns if c not in t13.columns]
print('new features:', newc, len(newc))
X = df[newc].astype(float).values
print('nonfinite in new block:', (~np.isfinite(X)).sum())
df[newc] = np.where(np.isfinite(X), X, 0.0)
print('final shape', df.shape)
path = agent_api.save_table(df, 'e018_nonlin.parquet')
print(path)


# ---- cell ----
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e013_union.parquet').copy()   # best table (val MAE 60.703)
base = set(df.columns)
def col(c, fill=0.0):
    return df[c].astype(float).fillna(fill).values if c in base else np.zeros(len(df))

m_day = col('sp364').sum() / max((df['sp364'].astype(float)>0).sum()*364.0, 1.0)
new = pd.DataFrame(index=df.index)
# concave transforms of heavy-tailed levels -> closer to conditional median under MAE
for c in ['sp28','sp84','sp364','sp28_rate','sp364_rate','wksp_mean','z_med4w_hist','z_mean_week_spend_all']:
    new['sq_'+c] = np.sqrt(np.clip(col(c),0,None))
    new['lg_'+c] = np.log1p(np.clip(col(c),0,None))
# empirical-Bayes shrunk expected next-4w spend (rate shrunk to global daily rate)
for w,k in [(84,56),(168,56),(364,112)]:
    new['exp_eb%d'%w] = 28.0*(col('sp%d'%w) + k*m_day)/(w+k)
zr = np.clip(col('z_zero_rate_hist',0.5),0,1)
new['expz84']  = new['exp_eb84'].values *(1-zr)
new['expz364'] = new['exp_eb364'].values*(1-zr)
# interactions of level x cadence/recency
dsl = np.clip(col('days_since_last',999),0,728)
new['i_rec_level'] = col('sp28')*np.exp(-dsl/28.0)
new['i_trips_lvl'] = col('sp28_rate')*col('trips28')
new['i_act_lvl']   = col('sp84_rate')*col('nact84')
new['i_med_max']   = col('z_med4w_hist')*col('z_max4w_hist')
new['i_ew_mom']    = col('sp28_rate')*col('m_sp28_56')
new['i_wk_cv']     = col('wksp_mean')/(col('wksp_std')+1.0)
new['lvl_blend']   = 0.5*new['exp_eb84'].values + 0.25*new['exp_eb364'].values + 0.25*(col('z_mean_week_spend_all')*28.0)
new = new.replace([np.inf,-np.inf], np.nan)
df = pd.concat([df, new], axis=1)
print('added', new.shape[1], 'features; total cols', df.shape[1])
print('nonfinite in new block:', (~np.isfinite(new.values)).sum())
path = agent_api.save_table(df, 'e018_nonlin.parquet')
print(path)
