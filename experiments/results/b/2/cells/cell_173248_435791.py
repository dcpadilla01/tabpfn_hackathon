import pandas as pd, numpy as np, agent_api

base = agent_api.load_saved('e015_market_ctx.parquet')
print('base shape', base.shape)
cols = [c for c in base.columns if c not in ('household_key','snapshot_day')]
# drop day_idx and x_* long-run features
drop_cols = [c for c in cols if c == 'day_idx' or c.startswith('x_')]
print('dropping', len(drop_cols), drop_cols[:25])
T = base.drop(columns=drop_cols).copy()
# add hinge features on log1p of key spend columns
hgs=[]
for src in ['spend_84','spend_28','fwd28_mean','wk_avg_84','spend_56']:
    lv = np.log1p(np.maximum(T[src].fillna(0).to_numpy(float),0))
    qs = np.quantile(lv, [0.15,0.3,0.45,0.6,0.75,0.9])
    for j,q in enumerate(qs):
        T[f'hg_{src}_{j}'] = np.maximum(lv-q, 0); hgs.append(f'hg_{src}_{j}')
print('final shape', T.shape, 'n_feat', T.shape[1]-2)
print('rows per day:'); print(T.snapshot_day.value_counts().sort_index().to_string())
assert T.shape[1]-2 <= 500
p = agent_api.save_table(T, 'e018_hinge_prune')
print('saved', p)