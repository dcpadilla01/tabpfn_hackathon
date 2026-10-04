import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values

def loso(Xs, lam, tag):
    errs=[]; pms=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean()); pms.append(p.mean())
    print(f"{tag} lam={lam}: folds=" + " ".join(f"{e:.0f}" for e in errs) + f"  mean={np.mean(errs):.2f}  predmeans={np.round(pms,0)}")

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
Q=np.where(np.isnan(F), mu, (F-mu)/sd)
M=(np.where(np.isnan(F), mu, F)-mu)/sd
for lam in [100, 1000, 3000, 10000]:
    loso(Q, lam, "quirk   ")
    loso(M, lam, "meanfill")