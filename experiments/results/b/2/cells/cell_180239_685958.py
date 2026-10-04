import pandas as pd, numpy as np, agent_api as A
train_days = sorted(set(A.snapshot_days()['train']))
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats0 = [c for c in base.columns if c not in ('household_key','snapshot_day')]
y = df.future_spend_4w.values; days = df.snapshot_day.values
F = df[feats0].astype(float).values

def loso(Xs, lam, tag):
    errs=[]
    for s in train_days:
        m = days!=s
        b = np.linalg.solve(Xs[m].T@Xs[m]+lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
        p = np.clip(Xs[~m]@b,0,None)
        errs.append(np.abs(p-y[~m]).mean())
    return np.mean(errs)

mu=np.nanmean(F,0); sd=np.nanstd(F,0)+1e-9
one = np.ones((len(y),1))
Mint = np.hstack([(np.where(np.isnan(F), mu, F)-mu)/sd, one])
Qint = np.hstack([np.where(np.isnan(F), mu, (F-mu)/sd), one])
for lam in [100, 300, 1000, 3000]:
    print(f"lam={lam}: meanfill+int={loso(Mint,lam,'m'):.2f}  quirk+int={loso(Qint,lam,'q'):.2f}")