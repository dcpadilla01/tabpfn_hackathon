import pandas as pd, numpy as np, agent_api as A

train_days = set(A.snapshot_days()['train'])
tt = A.train_targets()
base = A.load_saved('e019_final.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'], how='inner')
KEYS = ['household_key','snapshot_day']
feats0 = [c for c in base.columns if c not in KEYS]
y = df.future_spend_4w.values
days = df.snapshot_day.values
is_tr = np.array([d in train_days for d in days])

def stdize(Xtr, Xall):
    mu = np.nanmean(Xtr,0); sd = np.nanstd(Xtr,0)+1e-9
    Z = np.where(np.isnan(Xall), mu, Xall)
    return (Z-mu)/sd

def cv(Xall, lams=(30,100,300)):
    out=[]
    Xtr_raw = Xall[is_tr]
    for lam in lams:
        errs=[]
        Xs = stdize(Xtr_raw, Xall)
        for s in sorted(train_days):
            m = days!=s
            b = np.linalg.solve(Xs[m].T@Xs[m] + lam*np.eye(Xs.shape[1]), Xs[m].T@y[m])
            pv = np.clip(Xs[~m]@b, 0, None)
            errs.append(np.abs(pv-y[~m]).mean())
        out.append((lam, np.mean(errs)))
    return out

F = df[feats0].astype(float).values
print("BASE:", [(l, round(m,3)) for l,m in cv(F)])

# C1/C2 winsorize
for lo,hi in [(0.01,0.99),(0.005,0.995),(0.02,0.98)]:
    q1 = np.nanquantile(F[is_tr], lo, axis=0); q2 = np.nanquantile(F[is_tr], hi, axis=0)
    W = np.clip(F, q1, q2)
    print(f"WINS{lo}: ", [(l, round(m,3)) for l,m in cv(W)])
