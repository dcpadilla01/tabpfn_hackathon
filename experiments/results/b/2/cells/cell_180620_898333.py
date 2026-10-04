import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
LAM=1000
def loso(Xraw, tag):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    Xs=np.hstack([(np.where(np.isnan(Xraw),mu,Xraw)-mu)/sd, np.ones((len(Xraw),1))])
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        errs.append(np.abs(np.clip(Xs[~m]@b,0,None)-y[~m]).mean())
    print(f"{tag:40s} {np.mean(errs):.2f}")

mkt_fwd = df.groupby('snapshot_day').fwd28_mean.transform('mean').values
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
Inter=np.column_stack([df[t_].astype(float).values*df[d_].astype(float).values for t_ in top for d_ in demo])
# recency / zero-streak hinges (zero-spend regime)
rec = df.recency.astype(float).values; zs = df.zero_streak.astype(float).values
Hr = np.column_stack([np.maximum(0, rec-k) for k in (14,28,56,84,112)] +
                     [np.maximum(0, zs-k) for k in (0,1,2,4,8)] +
                     [(rec>28).astype(float), (rec>56).astype(float), (rec>84).astype(float)])
E1 = np.hstack([F, Inter]); loso(E1, "E1: e019 + demo interactions")
E2 = np.hstack([F, Inter, mkt_fwd[:,None]]); loso(E2, "E2: + demo inter + mkt_fwd")
E3 = np.hstack([F, Hr]); loso(E3, "E3: + recency/streak hinges")
E4 = np.hstack([F, Inter, Hr, mkt_fwd[:,None]]); loso(E4, "E4: ALL combined")
E5 = np.hstack([F, Hr, mkt_fwd[:,None]]); loso(E5, "E5: hinges + mkt_fwd")