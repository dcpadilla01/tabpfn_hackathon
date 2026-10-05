import pandas as pd, numpy as np
f3 = agent_api.load_saved('feats_v3.parquet')
print('feats_v3', f3.shape)
print(f3.columns.tolist())
print(f3.groupby('snapshot_day').size())
tt = agent_api.train_targets()
y = tt.future_spend_4w
print('targets', tt.shape)
print(y.describe())
print('zero frac', (y==0).mean())
print('quantiles', y.quantile([.5,.75,.9,.95,.99,.995]).to_dict())
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape)
print('na frac top:', m.isna().mean().sort_values(ascending=False).head(8).to_dict())
num = [c for c in m.columns if c not in ('household_key','future_spend_4w') and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num+['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print('top |corr| with target:')
print(cor.head(25))
for c in ['exp4w_blend','lag0_spend','lag1_spend','lag2_spend']:
    if c in m.columns:
        print(c, 'MAE vs target:', (m[c].clip(lower=0)-m['future_spend_4w']).abs().mean().round(2))
