import pandas as pd, numpy as np, agent_api as A
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values
F = df[feats0].astype(float).values
mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
M=(np.where(np.isnan(F), mu, F)-mu)/sd
XtX = M.T@M; Xty = M.T@y
for lam in [100, 300, 1000]:
    b = np.linalg.solve(XtX+lam*np.eye(M.shape[1]), Xty)
    p = M@b
    print(f"lam={lam}: pred mean={p.mean():.2f} pred std={p.std():.2f} corr={np.corrcoef(p,y)[0,1]:.3f} MAE={np.abs(np.clip(p,0,None)-y).mean():.2f}")
print("y: mean", y.mean().round(1), "std", y.std().round(1))
# check per-column scale of XtX diag
print("XtX diag min/med:", np.diag(XtX).min().round(2), np.median(np.diag(XtX)).round(2))
# condition-ish: top eigenvalue
ev = np.linalg.eigvalsh(XtX)
print("eig min/max:", ev.min().round(3), ev.max().round(1))