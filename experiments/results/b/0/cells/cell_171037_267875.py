import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

mac = agent_api.load_saved('macro.parquet').reset_index()
tt = agent_api.train_targets()
df = agent_api.load_saved('e018_basestab.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')

# 1) macro values by day
md = mac.groupby('snapshot_day').mean(numeric_only=True)
print('macro by day:')
print(md.round(3).to_string())

# 2) day-level future spend (train days only; view capped at 459)
v = agent_api.snapshot(459)
txn = v.transactions
rows=[]
for d in sorted(df.snapshot_day.unique()):
    if d>431: continue
    fut = txn[(txn.day>d)&(txn.day<=d+28)]
    rows.append({'day':d, 'fut_per_hh': fut.sales_value.sum()/len(fut.household_key.unique()) if len(fut) else 0})
fut = pd.DataFrame(rows).set_index('day')
print('\nday-level future spend per hh:')
print(fut.round(2).to_string())
j = md.join(fut)
print('\ncorrelation of macro features with day-level future spend:')
print(j.corr()['fut_per_hh'].drop('fut_per_hh').round(3).sort_values())