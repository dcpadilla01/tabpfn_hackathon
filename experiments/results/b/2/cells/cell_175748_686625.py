import pandas as pd, numpy as np, agent_api as A

train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values; days = df.snapshot_day.values

F = df[feats0].astype(float).values
nan_frac = np.isnan(F).mean(0)
order = np.argsort(-nan_frac)
print("cols with NaN>0:", (nan_frac>0).sum(), "of", len(feats0))
for i in order[:25]:
    if nan_frac[i]>0: print(f"  {feats0[i]:22s} {nan_frac[i]:.3f}")

def cv(Xall, fill='quirk', lams=(100,300)):
    out=[]
    for lam in lams:
        errs=[]
        if fill=='quirk':
            mu=np.nanmean(Xall,0); sd=np.nanstd(Xall,0)+1e-9
            Xs=np.where(np.isnan(Xall), mu, (Xall-mu)/sd)
        else:
            mu=np.nanmean(Xall,0); sd=np.nanstd(Xall,0)+1e-9
            Xs=(np.where(np.isnan(Xall), mu, Xall)-mu)/sd
        for s in train_days:
            m=days!=s
            b=np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
            errs.append(np.abs(np.clip(Xs[~m]@b,0,None)-y[~m]).mean())
        out.append((lam, round(float(np.mean(errs)),3)))
    return out

print("quirk:", cv(F,'quirk'))
print("meanfill:", cv(F,'mean'))

# 0-fill + explicit indicators for cols with nan_frac>0.5%
ind_cols=[feats0[i] for i in range(len(feats0)) if 0.005<nan_frac[i]]
I = np.isnan(F[:, [feats0.index(c) for c in ind_cols]]).astype(float)
F2 = np.where(np.isnan(F), 0.0, F)  # careful: 0 in RAW units, then standardize
X2 = np.hstack([F2, I])
print(f"meanfill+ind ({I.shape[1]} inds):", cv(X2,'mean'))