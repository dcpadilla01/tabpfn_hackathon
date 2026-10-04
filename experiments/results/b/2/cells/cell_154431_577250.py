
import agent_api
import pandas as pd, numpy as np

names = ['e001_recent_spend','e004_temporal','e005_longrun','e006_fwd_profile','e008_fwd_calendar',
         'e009_demo','e012_basket_shape','e009_demo_mkt','e011_discounts']
for n in names:
    try:
        t = agent_api.load_saved(n + '.parquet')
        print('==', n, t.shape)
        print(list(t.columns))
        print()
    except Exception as e:
        print(n, 'ERR', repr(e))

tt = agent_api.train_targets()
ycol = agent_api.TARGET
print('targets', tt.shape, tt.columns.tolist())
y = tt[ycol]
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2).to_string())
print('zero share', round(float((y==0).mean()),3))
print(tt.groupby('snapshot_day')[ycol].agg(['mean','median']).round(1).to_string())

t12 = agent_api.load_saved('e012_basket_shape.parquet')
m = t12.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
num = [c for c in m.columns if c not in ('household_key','snapshot_day',ycol) and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m[ycol])
print('--- correlations with target (sorted) ---')
print(cor.sort_values().round(3).to_string())
