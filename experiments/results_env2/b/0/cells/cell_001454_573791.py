
import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
e13 = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
m = e13.merge(tt, on=['household_key','snapshot_day'])
tr = m[m.snapshot_day<=431]; va = m[m.snapshot_day>431]
ytr = tr.future_spend_4w.values; yva = va.future_spend_4w.values
feats = [c for c in e13.columns if c not in ('household_key','snapshot_day')]
def prep(df):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = X[c].astype('category').cat.codes.replace(-1, np.nan)
    return X.astype(float)
Xtr = prep(tr); Xva = prep(va)
med = Xtr.median(); Xtr = Xtr.fillna(med); Xva = Xva.fillna(med)
print('any NaN after fill:', Xtr.isna().sum().sum(), Xva.isna().sum().sum())
mu, sd = Xtr.mean(), Xtr.std().replace(0,1)
Xtr_s = ((Xtr-mu)/sd).values; Xva_s = ((Xva-mu)/sd).values
print('nan in Xtr_s:', np.isnan(Xtr_s).sum(), 'nan Xva_s:', np.isnan(Xva_s).sum())
lam=30
X1 = np.c_[np.ones(len(Xtr_s)), Xtr_s]
A = X1.T@X1 + lam*np.eye(X1.shape[1]); A[0,0]-=lam
print('A nan:', np.isnan(A).sum())
b = np.linalg.solve(A, X1.T@ytr)
print('b nan:', np.isnan(b).sum())
