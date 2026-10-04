import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e019_final.parquet')
tt = A.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged:", df.shape)

train_days = A.snapshot_days()['train']; val_days = A.snapshot_days()['validation']
tr = df[df.snapshot_day.isin(train_days)]
va = df[df.snapshot_day.isin(val_days)]
print("train rows:", len(tr), "val rows:", len(va))

feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
Xtr = tr[feats].astype(float).values; Xva = va[feats].astype(float).values
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)+1e-9
Xtr = np.where(np.isnan(Xtr), mu, (Xtr-mu)/sd); Xva = np.where(np.isnan(Xva), mu, (Xva-mu)/sd)

def fit_ridge(X, y, lam, w=None):
    if w is None:
        return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
    XtX = X.T@(X*w[:,None]); return np.linalg.solve(XtX + lam*np.eye(X.shape[1]), X.T@(w*y))

for lam in [3,10,30,100,300]:
    b = fit_ridge(Xtr, ytr, lam)
    pv = np.clip(Xva@b, 0, None)
    print(f"lam={lam}: val MAE={np.abs(pv-yva).mean():.3f}  trainMAE={np.abs(Xtr@b-ytr).mean():.3f}")
