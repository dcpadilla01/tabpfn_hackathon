
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
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    X = X.astype(np.float64)
    return X

Xtr, Xva = prep(tr), prep(va)
allnan = Xtr.columns[Xtr.isna().all()].tolist()
const = [c for c in Xtr.columns if Xtr[c].nunique(dropna=True)<=1]
print('all-nan cols:', allnan)
print('constant cols:', const[:20], '... total', len(const))
keep = [c for c in cols if c not in allnan and c not in const]
print('kept', len(keep), 'of', len(cols))
Xtr, Xva = Xtr[keep].values, Xva[keep].values
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0)
med = np.where(np.isnan(med), 0, med)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
Xtr = np.nan_to_num(Xtr); Xva = np.nan_to_num(Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)+1e-9
Xtr_s, Xva_s = (Xtr-mu)/sg, (Xva-mu)/sg

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
print('INNER (431):')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s[~iv], ytr[~iv], lam)
    print(' lam', lam, 'inner MAE', round(np.abs(Xtr_s[iv]@w-ytr[iv]).mean(),2))
print('FULL->VAL:')
for lam in [0.1,1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    p = Xva_s@w
    print(' lam', lam, 'val MAE', round(np.abs(p-yva).mean(),2), 'R2', round(1-((p-yva)**2).mean()/yva.var(),3))
