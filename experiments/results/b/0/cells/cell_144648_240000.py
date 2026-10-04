
import numpy as np, pandas as pd

for name in ['mkt_demo','rfm28','rich_behavioral','season']:
    df = agent_api.load_saved(name + '.parquet')
    print('==', name, df.shape)
    print(list(df.columns))
    print()

tt = agent_api.train_targets()
y = tt.future_spend_4w
print('train rows', len(tt), 'mean', round(y.mean(),2), 'median', y.median(), 'zero share', round((y==0).mean(),3))
print('MAE median-pred', round((y - y.median()).abs().mean(),3), 'MAE mean-pred', round((y - y.mean()).abs().mean(),3))

rfm = agent_api.load_saved('rfm28.parquet')
m = tt.merge(rfm, on=['household_key','snapshot_day'], how='left')
for c in rfm.columns:
    if c not in ('household_key','snapshot_day'):
        print('MAE', c, '=', round((y - m[c]).abs().mean(),3))

base = agent_api.load_saved('mkt_demo.parquet')
mm = tt.merge(base, on=['household_key','snapshot_day'], how='left')
num = [c for c in base.columns if c not in ('household_key','snapshot_day') and pd.api.types.is_numeric_dtype(mm[c])]
corrs = mm[num].corrwith(mm.future_spend_4w).sort_values()
print('\nTOP |corr| with target (mkt_demo):')
print(corrs.reindex(corrs.abs().sort_values(ascending=False).index).head(15))

v = agent_api.snapshot()
print('\ntx rows<=459:', v.transactions.shape)
print('n departments:', v.products.department.nunique())
print('n commodities:', v.products.commodity_desc.nunique())
