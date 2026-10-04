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
