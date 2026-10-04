import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
print('hh type:', type(v.households))
try:
    print(v.households.head())
except Exception as e:
    print('hh head fail', repr(e))
print('v.day', v.day, 'v.week', v.week)
tx = v.transactions
print('tx', tx.shape)
print(tx[['sales_value','quantity']].describe().to_string())
print('neg share', float((tx.sales_value<0).mean()), 'zero share', float((tx.sales_value==0).mean()))
d = tx[tx.day>459-56]
bs = d.groupby(['household_key','basket_id'])['sales_value'].sum()
print('baskets/hh 56d:'); print(bs.groupby('household_key').size().describe().to_string())
t = agent_api.train_targets()
print('targets', t.shape)
print(t.future_spend_4w.describe().to_string())
print('zero share', float((t.future_spend_4w==0).mean()))
print(t.future_spend_4w.quantile([.5,.75,.9,.95,.99]).to_string())
for p in ['e011_rank.parquet']:
    try:
        e = agent_api.load_saved(p); print('loaded', p, e.shape); print(list(e.columns)[:40])
    except Exception as ex: print('fail', p, repr(ex))
print(agent_api.snapshot_days())
