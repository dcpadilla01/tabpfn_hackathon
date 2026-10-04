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
