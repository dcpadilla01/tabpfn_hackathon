import pandas as pd, numpy as np, agent_api as A

train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
has_nan = np.isnan(F).any(1)
print("rows with any NaN:", has_nan.mean().round(3), " mean y:", y[has_nan].mean().round(1), " vs no-NaN:", y[~has_nan].mean().round(1), " overall:", y.mean().round(1))
# y distribution by NaN status
for nm, m in [('nan',has_nan),('non',~has_nan)]:
    yy=y[m]; print(nm, "p50/75/90:", np.percentile(yy,[50,75,90]).round(1), "frac y=0:", (yy==0).mean().round(3))

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
def fit_pred(Xs):
    preds=np.zeros(len(y))
    for s in train_days:
        m=days!=s
        b=np.linalg.solve(Xs[m].T@Xs[m]+100*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        preds[~m]=np.clip(Xs[~m]@b,0,None)
    return preds
Q=np.where(np.isnan(F), mu, (F-mu)/sd); pq=fit_pred(Q)
M=(np.where(np.isnan(F), mu, F)-mu)/sd; pm=fit_pred(M)
for nm,m in [('nan',has_nan),('non',~has_nan)]:
    print(nm, "MAE quirk:", np.abs(pq[m]-y[m]).mean().round(2), " MAE meanfill:", np.abs(pm[m]-y[m]).mean().round(2),
          " mean pred q/m:", pq[m].mean().round(1), pm[m].mean().round(1), " mean y:", y[m].mean().round(1))
print("overall corr(pq,pm):", np.corrcoef(pq,pm)[0,1].round(3))
# where does quirk win? error by y-quantile
qb=np.q= pd.qcut(y, 5, duplicates='drop')
print(pd.DataFrame({'y_bin':qb,'eq':np.abs(pq-y),'em':np.abs(pm-y)}).groupby('y_bin',observed=True)[['eq','em']].mean().round(2))