import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
LAM=1000
def loso(Xs, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    print(f"{tag:38s} {np.mean(errs):.2f}")
    return np.mean(errs)
def prep(Xraw):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    return np.hstack([(np.where(np.isnan(Xraw), mu, Xraw)-mu)/sd, np.ones((len(Xraw),1))])

# market-level forward proxy per snapshot
mkt_fwd = df.groupby('snapshot_day').fwd28_mean.transform(lambda s: s.mean())
mkt_fwd84 = df.groupby('snapshot_day').wk_avg_84.transform(lambda s: s.mean())
Fm = np.hstack([F, mkt_fwd.values[:,None], mkt_fwd84.values[:,None]])
loso(prep(Fm), "D1 +mkt_fwd, mkt_fwd84")

# prune: full-train ridge, keep top-K by |std coef|
mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
Xs=np.hstack([(np.where(np.isnan(F),mu,F)-mu)/sd, np.ones((len(F),1))])
b_full = np.linalg.solve(Xs[days!=431].T@Xs[days!=431]+LAM*np.eye(Xs.shape[1]), Xs[days!=431].T@y[days!=431])
imp = np.abs(b_full[:-1])
order = np.argsort(-imp)
for K in [60, 100, 150, 200]:
    keep = order[:K]
    loso(prep(F[:, keep]), f"D2 prune top-{K}")

# rank-transform key spend features (within snapshot)
key = ['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','spend_364','wk_avg_4']
R = np.hstack([df.groupby('snapshot_day')[k].rank(pct=True).values[:,None] for k in key])
loso(prep(np.hstack([F, R])), "D3 +8 pct-rank features")
loso(prep(np.hstack([F[:, order[:200]], R])), "D4 top-200 + ranks")