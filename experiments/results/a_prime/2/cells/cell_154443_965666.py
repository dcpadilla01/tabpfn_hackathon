
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day.isin(sd['train'])]
va = m[m.snapshot_day.isin(sd['validation'])]
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]

def prep(df):
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
    return X.astype(np.float64).values

Xtr, Xva = prep(tr), prep(va)
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
Xtr_s, Xva_s = (Xtr-mu)/sg, (Xva-mu)/sg

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
Xa, ya, Xb, yb = Xtr_s[~iv], ytr[~iv], Xtr_s[iv], ytr[iv]
print('INNER (431) ridge:')
for lam in [1e-2,1e-1,1,10,100,1000]:
    w = rfit(Xa, ya, lam)
    print(' lam', lam, 'inner MAE', round(np.abs(Xb@w-yb).mean(),2))

print('\nFULL-train fit -> validation MAE (proxy for the harness model):')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    print(' lam', lam, 'val MAE', round(np.abs(Xva_s@w-yva).mean(),2), 'R2', round(1-((Xva_s@w-yva)**2).mean()/yva.var(),3))
