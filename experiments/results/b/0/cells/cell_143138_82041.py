import agent_api, numpy as np, pandas as pd
tt = agent_api.train_targets()
snap = agent_api.snapshot()
tr = snap.transactions

# per (hh, snapshot) last-28d spend and trips, computed for train snapshots
res = []
for sd in agent_api.snapshot_days()['train']:
    v = agent_api.snapshot(as_of_day=sd)
    t = v.transactions
    w = t[t.day > sd-28].groupby('household_key').agg(s=('sales_value','sum'), n=('basket_id','nunique'), last_day=('day','max'))
    idx = pd.DataFrame(index=v.households)
    idx = idx.join(w).fillna({'s':0,'n':0})
    idx['recency'] = sd - idx['last_day']
    idx['snapshot_day']=sd
    res.append(idx[['snapshot_day','s','n','recency']])
f = pd.concat(res).reset_index().rename(columns={'index':'household_key'})
m = tt.merge(f, on=['household_key','snapshot_day'])
print(m[['future_spend_4w','s','n','recency']].corr().round(3))
# by snapshot
print(m.groupby('snapshot_day')[['future_spend_4w','s']].mean().round(1))
# spend28 vs target scatter quantiles
m['bin'] = pd.qcut(m.s, 10, duplicates='drop')
print(m.groupby('bin').agg(t=('future_spend_4w','mean'), s=('s','mean'), n=('n','mean')).round(1))
