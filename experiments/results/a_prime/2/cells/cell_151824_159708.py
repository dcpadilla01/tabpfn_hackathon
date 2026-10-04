import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
base = agent_api.load_saved('weekly_history.parquet')
mk = agent_api.load_saved('marketing_exposure.parquet')
print('base', base.shape, 'mkt', mk.shape)
mkt_cols = [c for c in mk.columns if c not in ('household_key','snapshot_day')]
print('mkt cols:', mkt_cols)
m = base.merge(mk, on=['household_key','snapshot_day'], how='left', suffixes=('','_mkt'))
m = m.loc[:, ~m.columns.duplicated()]
print('merged:', m.shape)
print('NaN rate in mkt cols:', round(float(m[mkt_cols].isna().mean().mean()),3))
p = agent_api.save_table(m, 'e008_main_plus_marketing')
print('saved:', p)