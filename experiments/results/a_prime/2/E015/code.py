import agent_api, pandas as pd, numpy as np, re
t = agent_api.load_saved('e009_ewma_longlags.parquet')
print('shape', t.shape)
print(t.dtypes.value_counts().to_string())
print('has index:', 'index' in t.columns)
num = t.select_dtypes(include=[np.number])
print('n numeric:', num.shape[1])
neg = [c for c in num.columns if num[c].min() < 0]
print('n neg cols:', len(neg)); print(neg[:40])
naf = num.isna().mean()
print('anyNaN:', int((naf>0).sum()), 'gt30%NaN:', int((naf>0.3).sum()))

tt = agent_api.train_targets()
print('tt shape', tt.shape)
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median'])
g['zero'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s:(s==0).mean())
print(g.to_string())
print(tt['future_spend_4w'].describe().to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
cats = [c for c in t.columns if str(t[c].dtype)=='category' or t[c].dtype==bool]
print('categorical/bool cols:', cats)
for c in cats: print(c, t[c].value_counts(dropna=False).head(3).to_dict())

feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
print('n feat cols', len(feat_cols))

def ridge_eval(X, y, tr_mask, va_mask, alphas=(30,100,300,1000,3000)):
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0)+1e-9
    Xs = (X-mu)/sd
    Xtr, ytr = Xs[tr_mask], y[tr_mask]
    G = Xtr.T@Xtr; b = Xtr.T@ytr
    best=None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(G.shape[0]), b)
        pred = Xs[va_mask]@w
        mae = np.abs(pred-y[va_mask]).mean()
        if best is None or mae<best[0]: best=(mae,a,w)
    return best

train_snaps = sorted(tt.snapshot_day.unique())
print('train snaps', train_snaps)
m_tr = df.snapshot_day<=403; m_va = df.snapshot_day==431
X = df[feat_cols].copy()
for c in cats:
    X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
X = X.astype(np.float64).values
y = df.future_spend_4w.values
mae,a,w = ridge_eval(X, y, m_tr.values, m_va.values)
print('internal holdout snap431 MAE %.3f alpha %d'%(mae,a))
m_tr2 = df.snapshot_day!=403; m_va2 = df.snapshot_day==403
mae2,a2,_ = ridge_eval(X, y, m_tr2.values, m_va2.values)
print('internal holdout snap403 MAE %.3f alpha %d'%(mae2,a2))
for col in ['spend_28','tlag_mean','ewma_4']:
    if col in df.columns:
        print(col, 'MAE431 %.2f'%np.abs(df.loc[m_va,col].values-y[m_va.values]).mean())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
cats = [c for c in feat_cols if str(t[c].dtype)=='category' or t[c].dtype==bool]

def prep(df):
    X = df[feat_cols].copy()
    for c in cats:
        X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(np.float64)
    return X

def ridge_eval(X, y, tr_idx, va_idx, alphas=(30,100,300,1000,3000)):
    Xtr_raw = X[tr_idx]
    mu = np.nanmean(Xtr_raw,0); 
    Xf = np.where(np.isnan(X), 0.0, X-mu)   # fill NaN with mean -> 0 after centering
    sd = Xf[tr_idx].std(0)+1e-9
    Xs = Xf/sd
    Xtr, ytr = Xs[tr_idx], y[tr_idx]
    G = Xtr.T@Xtr; b = Xtr.T@ytr
    best=None
    for a in alphas:
        w = np.linalg.solve(G + a*np.eye(G.shape[0]), b)
        mae = np.abs(Xs[va_idx]@w - y[va_idx]).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

X = prep(df).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int)
m_va = (d==431).values; m_tr = (d<=403).values
print('holdout 431: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr,m_va))
m_va2=(d==403).values; m_tr2=(d<=375).values
print('holdout 403: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr2,m_va2))
m_va3=(d==375).values; m_tr3=(d<=347).values
print('holdout 375: MAE %.3f alpha %d'%ridge_eval(X,y,m_tr3,m_va3))
# log1p target ridge
ylog = np.log1p(y)
mae,a = ridge_eval(X,ylog,m_tr,m_va)
pred_log = None
mu = np.nanmean(X[m_tr],0); Xf=np.where(np.isnan(X),0,X-mu); sd=Xf[m_tr].std(0)+1e-9; Xs=Xf/sd
G=Xs[m_tr].T@Xs[m_tr]; b=Xs[m_tr].T@ylog[m_tr]
w=np.linalg.solve(G+a*np.eye(G.shape[0]),b)
pl = np.expm1(Xs[m_va]@w)
print('log-target holdout 431: MAE %.3f alpha %d'%(np.abs(pl-y[m_va]).mean(),a))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
print('nan count', np.isnan(Xv).sum())
inf_mask = np.isinf(Xv)
print('inf count', inf_mask.sum())
cols_inf = [feat_cols[j] for j in range(len(feat_cols)) if inf_mask[:,j].any()]
print('cols with inf:', cols_inf)
y = df.future_spend_4w.values.astype(float)
print('y nan', np.isnan(y).sum())
# check per column nan fraction >0.9
naf = np.isnan(Xv).mean(0)
print('cols >90% nan:', [feat_cols[j] for j in range(len(feat_cols)) if naf[j]>0.9])


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values
m_tr = d<=403; m_va = d==431
mu = np.nanmean(Xv[m_tr],0)
print('mu nan count', np.isnan(mu).sum())
bad = [feat_cols[j] for j in range(len(feat_cols)) if np.isnan(mu[j])]
print('all-nan-in-train cols:', bad)
Xf = np.where(np.isnan(Xv), 0.0, Xv-mu)
sd = Xf[m_tr].std(0)+1e-9
print('sd nan', np.isnan(sd).sum(), 'sd zero', (sd<=1e-9).sum())
Xs = Xf/sd
print('Xs nan', np.isnan(Xs).sum(), 'inf', np.isinf(Xs).sum())
G = Xs[m_tr].T@Xs[m_tr]
print('G nan', np.isnan(G).sum(), 'G inf', np.isinf(G).sum())
print('max abs Xs', np.nanmax(np.abs(Xs[m_tr])))
big = [feat_cols[j] for j in range(len(feat_cols)) if np.nanmax(np.abs(Xs[m_tr][:,j]))>1e6]
print('huge cols:', big[:10])


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values

def ridge_eval(Xv, y, tr, va, alphas=(30,100,300,1000,3000)):
    mu = np.nanmean(Xv[tr],0)
    Xf = np.where(np.isnan(Xv),0.0,Xv-mu)
    sd = Xf[tr].std(0)+1e-9; sd[sd<1e-9]=1.0
    Xs = Xf/sd
    G = Xs[tr].T@Xs[tr]; b = Xs[tr].T@y[tr]
    best=None
    for a in alphas:
        w = np.linalg.solve(G+a*np.eye(G.shape[0]), b)
        mae = np.abs(Xs[va]@w - y[va]).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

print('E009 holdout431: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=403,d==431))
print('E009 holdout403: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=375,d==403))
print('E009 holdout375: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=347,d==375))
# single best features
res=[]
for j,c in enumerate(feat_cols):
    mae,_ = ridge_eval(Xv[:,[j]],y,d<=403,d==431,alphas=(100,))
    res.append((mae,c))
res.sort()
print('best single feats:', [(round(m,2),c) for m,c in res[:12]])


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values

def ridge_eval(Xv, y, tr, va, alphas=(30,100,300,1000,3000), return_w=False):
    mu = np.nanmean(Xv[tr],0)
    Xc = Xv-mu
    Xf = np.where(np.isfinite(Xc), Xc, 0.0)
    sd = Xf[tr].std(0); sd[~np.isfinite(sd)|(sd<1e-9)]=1.0
    Xs = Xf/sd
    G = Xs[tr].T@Xs[tr]; b = Xs[tr].T@y[tr]
    best=None
    for a in alphas:
        w = np.linalg.solve(G+a*np.eye(G.shape[0]), b)
        mae = np.abs(Xs[va]@w - y[va]).mean()
        if best is None or mae<best[0]: best=(mae,a,w if return_w else None)
    return best

print('E009 holdout431: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=403,d==431)[:2])
print('E009 holdout403: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=375,d==403)[:2])
print('E009 holdout375: MAE %.3f alpha %d'%ridge_eval(Xv,y,d<=347,d==375)[:2])
# log1p target
ylog = np.log1p(y)
mae,a,w = ridge_eval(Xv,ylog,d<=403,d==431,return_w=True)
mu=np.nanmean(Xv[d<=403],0); Xc=Xv-mu; Xf=np.where(np.isfinite(Xc),Xc,0)
sd=Xf[d<=403].std(0); sd[~np.isfinite(sd)|(sd<1e-9)]=1
pl = np.expm1(np.where(np.isfinite((Xf/sd)[d==431]@w), (Xf/sd)[d==431]@w, 0))
print('log-target holdout431: MAE %.3f alpha %d'%(np.abs(pl-y[d==431]).mean(),a))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day','index')]
X = df[feat_cols].copy()
for c in X.columns:
    if str(X[c].dtype)=='category' or X[c].dtype==bool:
        X[c]=X[c].astype('category').cat.codes.replace(-1,np.nan)
Xv = X.astype(np.float64).values
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values
va = d==431; tr = d<=403

# baselines on holdout 431
print('const median %.2f'%np.abs(np.median(y[tr])-y[va]).mean())
meds = {s:np.median(y[d==s]) for s in np.unique(d[tr])}
print('per-snap median %.2f'%np.abs(df.loc[va,'snapshot_day'].map(meds).values-y[va]).mean())
print('raw tlag_mean %.2f'%np.abs(df.loc[va,'tlag_mean'].values-y[va]).mean())

def ridge_eval(Xv, y, tr, va, alphas, log_t=False, rank=False):
    if rank:
        Xv = Xv.copy()
        for j in range(Xv.shape[1]):
            col = Xv[:,j]; m=np.isfinite(col)
            if m.sum()>10:
                r = np.argsort(np.argsort(col[m]))
                Xv[m,j] = (r+0.5)/m.sum()*2-1
                Xv[~m,j] = 0.0
            else: Xv[:,j]=0.0
    yt = np.log1p(y) if log_t else y
    mu = np.nanmean(Xv[tr],0); Xc=Xv-mu; Xf=np.where(np.isfinite(Xc),Xc,0.0)
    sd=Xf[tr].std(0); sd[~np.isfinite(sd)|(sd<1e-9)]=1.0; Xs=Xf/sd
    G=Xs[tr].T@Xs[tr]; b=Xs[tr].T@yt[tr]
    best=None
    for a in alphas:
        w=np.linalg.solve(G+a*np.eye(G.shape[0]),b)
        p = Xs[va]@w
        if log_t: p=np.expm1(np.clip(p,0,20))
        mae=np.abs(p-y[va]).mean()
        if best is None or mae<best[0]: best=(mae,a)
    return best

alphas=(100,300,1000,3000,10000,30000,100000)
print('rank-ridge raw-tgt 431:', ridge_eval(Xv,y,tr,va,alphas))
print('rank-ridge log-tgt 431:', ridge_eval(Xv,y,tr,va,alphas,log_t=True))
print('plain-ridge log-tgt big-alpha 431:', ridge_eval(Xv,y,tr,va,alphas,log_t=True,rank=False))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values
va = d==431; tr = d<=403

hh_tr = set(df.loc[tr,'household_key']); hh_va = set(df.loc[va,'household_key'])
print('households: train rows %d, val rows %d, overlap %d, val-only %d'%(len(hh_tr),len(hh_va),len(hh_tr&hh_va),len(hh_va-hh_tr)))

tr_df = df.loc[tr]
hh_med = tr_df.groupby('household_key')['future_spend_4w'].median()
hh_cnt = tr_df.groupby('household_key')['future_spend_4w'].size()
va_df = df.loc[va].copy()
med_map = va_df.household_key.map(hh_med)
cnt_map = va_df.household_key.map(hh_cnt).fillna(0)
med_glob = np.median(y[tr])
va_df['hh_med'] = med_map.fillna(med_glob)
va_df['hh_shr'] = (med_map*cnt_map + med_glob*3.0)/(cnt_map+3.0)
print('global median MAE %.2f'%np.abs(med_glob-y[va]).mean())
print('per-household median MAE %.2f'%np.abs(va_df.hh_med-y[va]).mean())
print('shrunk(k=3) MAE %.2f'%np.abs(va_df.hh_shr-y[va]).mean())
tl = df.loc[va,'tlag_mean'].values
for w in [0.3,0.5,0.7]:
    print('blend tlag+hh_med w=%.1f MAE %.2f'%(w, np.abs(w*tl+(1-w)*va_df.hh_med.values-y[va]).mean()))

print(df[['household_key','snapshot_day','index']].head(8).to_string())
print(df.groupby('snapshot_day')['index'].agg(['min','max']).head(4).to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
t['hh'] = t['household_key'].astype('category')
print(t.shape, t['hh'].nunique(), t['hh'].dtype)
p = agent_api.save_table(t, 'e015_hh_identity.parquet')
print(p)
