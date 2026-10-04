
import agent_api, pandas as pd, numpy as np

t12 = agent_api.load_saved('e012_basket_shape.parquet')
print(t12.shape)
print(list(t12.columns))
tt = agent_api.train_targets()
ycol = agent_api.TARGET
m = t12.merge(tt, on=['household_key','snapshot_day'], how='inner')
num = [c for c in m.columns if c not in ('household_key','snapshot_day',ycol) and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m[ycol]).sort_values()
print('--- 45 weakest correlations ---')
print(cor.head(45).round(3).to_string())
print('--- target dist ---')
y = tt[ycol]
print(y.describe(percentiles=[.5,.75,.9,.95,.99]).round(2).to_string())
print('zero share', round(float((y==0).mean()),3), 'mean', round(float(y.mean()),2), 'MAE of global median', round(float((y-y.median()).abs().mean()),2))
print('MAE of predict spend_28 (from e012):', )
m2 = m.dropna(subset=['spend_28'])
print(round(float((m2[ycol]-m2['spend_28']).abs().mean()),2))
print('MAE of 0.8*spend_28:', round(float((m2[ycol]-0.8*m2['spend_28']).abs().mean()),2))
print('snapdays', agent_api.snapshot_days())
