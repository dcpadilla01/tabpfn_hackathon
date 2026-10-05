import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

t = load_saved('e017_xsec_rank.parquet')
tt = train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
df = df[df['future_spend_4w'].notna()].copy()

def prep(d):
    X = d[feat_cols].copy()
    for c in feat_cols:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    return X.values.astype(float)

def fit_ridge(X, y, lam=100.0):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[sd==0]=1.0
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    A = Z.T@Z + lam*np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T@y), mu, sd

def apply_ridge(w, mu, sd, X):
    Z = (X-mu)/sd; Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    return Z@w

y = df['future_spend_4w'].values.astype(float)
days = df['snapshot_day'].values
X = prep(df)

tr = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375,403,431])
va = np.isin(days, [459,487,515,543])
for lam in [10, 30, 100, 300, 1000]:
    w, mu, sd = fit_ridge(X[tr], y[tr], lam=lam)
    pv = apply_ridge(w, mu, sd, X[va])
    print('lam %5d -> replica val MAE %.3f' % (lam, np.abs(pv-y[va]).mean()))

tr2 = np.isin(days, [95,123,151,179,207,235,263,291,319,347,375])
va2 = np.isin(days, [403,431])
w2, mu2, sd2 = fit_ridge(X[tr2], y[tr2])
pv2 = apply_ridge(w2, mu2, sd2, X[va2])
print('pseudo-val MAE (403,431): %.3f' % np.abs(pv2-y[va2]).mean())
