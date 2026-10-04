import pandas as pd, numpy as np, agent_api as A

t = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
tr_days = A.snapshot_days()
print(tr_days)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
df = t.merge(tt, on=['household_key','snapshot_day'])
tr = df[df.snapshot_day.isin(tr_days['train'])]
va = df[df.snapshot_day.isin(tr_days['validation'])]
print('train rows', len(tr), 'val rows', len(va))

def ridge_fit(X, y, lam):
    n, p = X.shape
    D = np.concatenate([np.zeros((1,1)), np.ones((1,p))], axis=1)  # not used
    A_ = np.hstack([np.ones((n,1)), X])
    R = A_.T @ A_ + lam*np.eye(p+1); R[0,0] -= lam  # don't penalize intercept
    return np.linalg.solve(R, A_.T @ y)

def prep(Xtr, Xva, mu=None, sd=None):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    return (Xtr-mu)/sd, (Xva-mu)/sd

Xtr_raw = tr[feats].fillna(0).values.astype(float)
Xva_raw = va[feats].fillna(0).values.astype(float)
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values

# lambda selection via leave-one-snapshot-out on train
lams = [1,3,10,30,100,300,1000,3000]
Xtr_s, _ = prep(Xtr_raw, Xtr_raw)
cv = {}
for lam in lams:
    errs = []
    for d in tr_days['train']:
        m = tr.snapshot_day.values != d
        Xi, yi = Xtr_s[m], ytr[m]
        mu = Xi.mean(0); sd = Xi.std(0); sd[sd==0]=1
        w = ridge_fit((Xi-mu)/sd, yi, lam)
        Xo = Xtr_s[~m]; Xo = (Xo-mu)/sd
        errs.append(np.mean(np.abs(Xo @ w[1:] + w[0] - ytr[~m])))
    cv[lam] = float(np.mean(errs))
print('LOSO-CV MAE by lambda:', {k: round(v,2) for k,v in cv.items()})
lam_best = min(cv, key=cv.get)
print('best lam', lam_best)

Xtr_s, Xva_s = prep(Xtr_raw, Xva_raw)
w = ridge_fit(Xtr_s, ytr, lam_best)
pv = Xva_s @ w[1:] + w[0]
print('LOCAL ridge on E003 feats: val MAE', round(float(np.mean(np.abs(pv-yva))),3), '(harness E003: 63.050)')
ptr = Xtr_s @ w[1:] + w[0]
print('train MAE', round(float(np.mean(np.abs(ptr-ytr))),3))
# clip at 0
print('val MAE clipped>=0:', round(float(np.mean(np.abs(np.clip(pv,0,None)-yva))),3))