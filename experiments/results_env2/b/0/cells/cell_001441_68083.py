
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
print('NaN cols in train:', Xtr.isna().sum()[Xtr.isna().sum()>0].to_dict())
print('NaN cols in val:', Xva.isna().sum()[Xva.isna().sum()>0].to_dict())
print('inf check:', np.isinf(Xtr.values).sum(), np.isinf(Xva.values).sum())
