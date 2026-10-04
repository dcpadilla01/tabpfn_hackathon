
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
    return X.astype(np.float64)

Xtr, Xva = prep(tr), prep(va)
# columns all-NaN in train but with values in validation -> the blowup source
trnan = Xtr.columns[Xtr.isna().all()].tolist()
vanan = Xva.columns[Xva.isna().all()].tolist()
print('all-NaN in TRAIN:', trnan)
print('all-NaN in VAL  :', vanan)
print('NaN in val but not train:', [c for c in cols if Xtr[c].notna().all()==False and c not in trnan][:10])

keep = [c for c in cols if c not in trnan]
Xtr, Xva = Xtr[keep].values, Xva[keep].values
ytr, yva = tr['future_spend_4w'].values, va['future_spend_4w'].values
med = np.nanmedian(Xtr, 0); med = np.where(np.isnan(med), 0, med)
Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
Xtr = np.nan_to_num(Xtr); Xva = np.nan_to_num(Xva)
mu, sg = Xtr.mean(0), Xtr.std(0)
dead = sg < 1e-12
print('zero-variance-after-fill cols:', keep.count if False else np.array(keep)[dead])
Xtr_s = np.clip((Xtr-mu)/np.where(dead,1,sg), -10, 10)[:, ~dead]
Xva_s = np.clip((Xva-mu)/np.where(dead,1,sg), -10, 10)[:, ~dead]
print('final feature count', Xtr_s.shape[1])

def rfit(X, y, lam): return np.linalg.solve(X.T@X + lam*np.eye(X.shape[1]), X.T@y)
iv = (tr.snapshot_day==431).values
print('INNER(431):', end=' ')
for lam in [1,10,100,1000]:
    w = rfit(Xtr_s[~iv], ytr[~iv], lam)
    print(f'lam{lam}:{np.abs(Xtr_s[iv]@w-ytr[iv]).mean():.2f}', end='  ')
print()
print('FULL->VAL :', end=' ')
for lam in [1,10,100,1000]:
    w = rfit(Xtr_s, ytr, lam)
    p = Xva_s@w
    print(f'lam{lam}:{np.abs(p-yva).mean():.2f}', end='  ')
print()
