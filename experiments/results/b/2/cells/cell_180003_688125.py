import pandas as pd, numpy as np, agent_api as A
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values
F = df[feats0].astype(float).values
mu=np.nanmean(F,0); sd=np.nanstd(F,0)
print("any sd==0:", (sd==0).sum(), " any mu nan:", np.isnan(mu).sum(), " any sd nan:", np.isnan(sd).sum())
M=(np.where(np.isnan(F), mu, F)-mu)/sd
print("M stats: mean", M.mean().round(3), "std", M.std().round(3), "max|.|", np.abs(M).max().round(2))
print("any nan in M:", np.isnan(M).sum(), " any inf:", np.isinf(M).sum())
# single-feature check
i = feats0.index('spend_84')
print("corr(spend_84, y):", np.corrcoef(np.nan_to_num(F[:,i]), y)[0,1].round(3))
b1 = np.linalg.solve(M[:,[i]].T@M[:,[i]]+1, M[:,[i]].T@y)
print("1-feat coef:", b1.round(3), " pred corr:", np.corrcoef(M[:,i]@b1, y)[0,1].round(3))
# full solve diagnostics
XtX = M.T@M
print("XtX diag range:", np.diag(XtX).min().round(3), np.diag(XtX).max().round(3))
b = np.linalg.solve(XtX+100*np.eye(M.shape[1]), M.T@y)
print("coef norm:", np.linalg.norm(b).round(3), " pred mean:", (M@b).mean().round(2), " y mean:", y.mean().round(2))
# maybe y has NaN?
print("y nan:", np.isnan(y).sum(), " y stats:", np.nanmin(y), np.nanmax(y), y.mean().round(1))