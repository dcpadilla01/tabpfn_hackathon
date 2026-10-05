
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e5 = agent_api.load_saved('e005_decay_gapcv.parquet')
print('saved rows', len(e5), 'snapshot days', sorted(e5.snapshot_day.unique()))
tt = agent_api.train_targets()
df = e5.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in e5.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
# internal CV: train on days <= 347, validate on 375/403/431
itr = df.snapshot_day<=347
iva = df.snapshot_day>=375
print('train rows', itr.sum(), 'pseudo-val rows', iva.sum())

X = df[feats].replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median()).values.astype(float)

def ridge(Xtr, ytr, Xva, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0)+1e-9
    A = (Xtr-mu)/sd; B = (Xva-mu)/sd
    A = np.c_[np.ones(len(A)), A]; B = np.c_[np.ones(len(B)), B]
    w = np.linalg.solve(A.T@A + lam*np.eye(A.shape[1]), A.T@ytr)
    return B@w

for name, Xf in [('raw', X), ('log1p', np.log1p(np.clip(X,0,None)))]:
    for lam in [1.0, 10.0, 100.0]:
        p = ridge(Xf[itr.values], y[itr.values], Xf[iva.values], lam)
        print('%-6s lam=%5.0f  pseudo-val MAE %.3f' % (name, lam, np.abs(p-y[iva.values]).mean()))
