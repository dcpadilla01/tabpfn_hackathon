
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
sd = agent_api.snapshot_days()
m = t.merge(tt, on=['household_key','snapshot_day'])
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = m[cols].select_dtypes(include=[np.number])
inf_cols = []
for c in num.columns:
    s = num[c]
    if np.isinf(s.values).any():
        inf_cols.append(c)
print('cols containing inf:', inf_cols)
for c in inf_cols:
    s = num[c].replace([np.inf,-np.inf], np.nan)
    print(c, 'max finite', s.max(), 'n_inf', np.isinf(num[c].values).sum())
print('\nindex col:', m['index'].describe().round(1).to_string())
big = []
for c in num.columns:
    s = num[c].replace([np.inf,-np.inf], np.nan).abs()
    mx = s.max()
    if mx > 1e6: big.append((c, mx))
print('\ncols with |max|>1e6:', big[:20])
