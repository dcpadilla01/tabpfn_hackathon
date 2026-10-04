import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values
one = np.ones((len(y),1))
LAM=1000

def loso(Xs, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+LAM*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    print(f"{tag:34s} {np.mean(errs):.2f}")
    return np.mean(errs)

def prep(Xraw):
    mu=np.nanmean(Xraw,0); sd=np.nanstd(Xraw,0)+1e-9
    return np.hstack([(np.where(np.isnan(Xraw), mu, Xraw)-mu)/sd, one])

loso(prep(F), "BASELINE e019 (intercept)")

# C1: winsorize 1/99
q1=np.nanquantile(F,0.01,0); q2=np.nanquantile(F,0.99,0)
loso(prep(np.clip(F,q1,q2)), "C1 winsorize 1/99")

# C2: missing indicators for the 20.8% NaN group (last-basket etc.)
nf = np.isnan(F).mean(0)
sel = (nf>0.05)&(nf<0.5)
I = np.isnan(F[:,sel]).astype(float)
X2 = np.hstack([np.where(np.isnan(F),0,F), I, one])
mu=np.nanmean(X2,0); sd=np.nanstd(X2,0)+1e-9
X2s=(X2-mu)/sd
loso(X2s, f"C2 zero-fill + {I.shape[1]} indicators")

# C3: extra hinges on more spend sources (6 quantiles each)
extra_src = ['spend_168','spend_364','spend_7','spend_14','spend_21','wk_avg_8','spend_life','spend_rate_life','spend_112','spend_252']
H=[]
for s_ in extra_src:
    v = df[s_].astype(float).values
    qs = np.nanquantile(v, [0.1,0.25,0.5,0.75,0.9,0.97])
    for q in qs:
        H.append(np.maximum(0.0, np.log1p(np.maximum(v,0)) - np.log1p(max(q,0))))
H=np.array(H).T
loso(prep(np.hstack([F,H])), f"C3 +{H.shape[1]} extra hinges")

# C4: interactions of top-8 raw features with demo ordinals
top=['spend_84','spend_28','wk_avg_8','fwd28_mean','ewma28_4w','spend_lag1','trips_84','spend_364']
demo=['size_ord','income_ord','age_ord','grp_ord','kid_ord']
Inter=[]
for t_ in top:
    v=df[t_].astype(float).values
    for d_ in demo:
        Inter.append(v*df[d_].astype(float).values)
Inter=np.array(Inter).T
loso(prep(np.hstack([F,Inter])), f"C4 +{Inter.shape[1]} demo interactions")