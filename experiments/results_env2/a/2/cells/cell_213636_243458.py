
import agent_api as api, pandas as pd, numpy as np

tt = api.train_targets()
snap = api.snapshot(459)
tr = snap.transactions

# lagged spend features per household
def lag_spend(as_of, w):
    d = tr[(tr.day > as_of - w) & (tr.day <= as_of)]
    return d.groupby('household_key').sales_value.sum()

hh = tt.household_key.unique()
res = {}
for as_of in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    sub = tt[tt.snapshot_day==as_of].set_index('household_key')
    for w in [28,56,112]:
        s = lag_spend(as_of, w).reindex(sub.index).fillna(0)
        res.setdefault(f'lag{w}', []).extend(s.values)
    res.setdefault('y', []).extend(sub.future_spend_4w.values)

for k in res:
    if k!='y':
        print(k, np.corrcoef(res[k], res['y'])[0,1], np.mean(np.abs(np.array(res[k])-np.array(res['y']))))
