import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w
for c in ['spend28','spend56','spend112','own_ly_spend4w','own_ly2_spend4w','season_lift','lt_spend','spend364']:
    s = m[c]
    print(f'{c:18s} mean {s.mean():9.1f} med {s.median():8.1f} p95 {s.quantile(.95):9.1f} max {s.max():12.1f} corr {s.corr(y):.3f}')
print()
print('spend28 describe:\n', m.spend28.describe())
print('spend56 describe:\n', m.spend56.describe())
# ratio check
r = (m.spend56/m.spend28.replace(0,np.nan)).dropna()
print('spend56/spend28 ratio quantiles:', r.quantile([.05,.25,.5,.75,.95]).round(2).values)
print('target quantiles:', y.quantile([.5,.9,.99]).values, 'max', y.max())