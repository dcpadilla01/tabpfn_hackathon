import numpy as np, pandas as pd
from agent_api import load_saved, train_targets

tt = train_targets()
KEY = ['household_key','snapshot_day']

def pseudo_mae(path, lam=100.0, holdout=(403,431)):
    t = load_saved(path).merge(tt, on=KEY, how='left')
    t = t[t['future_spend_4w'].notna()]
    fc = [c for c in t.columns if c not in KEY+['future_spend_4w']]
    X = t[fc].copy()
    for c in fc:
        if X[c].dtype == bool: X[c] = X[c].astype(float)
        elif not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.Categorical(X[c]).codes.astype(float)
    X = X.values.astype(float)
    y = t['future_spend_4w'].values.astype(float)
    d = t['snapshot_day'].values
    tr = ~np.isin(d, holdout); va = np.isin(d, holdout)
    mu = np.nanmean(X[tr],0); sd = np.nanstd(X[tr],0); sd[sd==0]=1
    Z = np.nan_to_num((X-mu)/sd)
    A = Z[tr].T@Z[tr] + lam*np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z[tr].T@y[tr])
    pv = Z[va]@w
    return np.abs(pv-y[va]).mean()

harness = {'e001_recent_behavior.parquet':63.574,'e010_lifecycle.parquet':62.702,
           'e011_discounts.parquet':62.651,'e015_full_superset.parquet':62.647,
           'e016_churn_gapratio.parquet':62.535,'e017_xsec_rank.parquet':62.526,
           'e009_ar_season.parquet':66.683,'e012_hh_target_enc.parquet':72.176}
res = []
for p,h in harness.items():
    m = pseudo_mae(p)
    res.append((m,h,p))
    print('%-32s pseudo %8.3f   harness %8.3f' % (p, m, h))
res.sort()
print('\npseudo rank:', [r[2][:4] for r in res])
print('harness rank:', [r[2][:4] for r in sorted(res, key=lambda r: r[1])])
print('corr check: pseudo best =', res[0][2], '| pseudo worst =', res[-1][2])
