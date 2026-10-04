import pandas as pd, numpy as np

t = agent_api.load_saved('e008_fwd_calendar.parquet')
tt = agent_api.train_targets()
print('e008 shape', t.shape)
print('dtype counts', t.dtypes.astype(str).value_counts().to_dict())
print('columns:'); print(list(t.columns))

y = tt['future_spend_4w']
print('\ntarget describe:')
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('zero frac', round(float((y==0).mean()),3))

m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('\nmerged', m.shape, 'nan cells', int(m.isna().sum().sum()))

feat_cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(m[c]) and m[c].dtype != bool]
bool_cols = [c for c in feat_cols if m[c].dtype == bool]
cat_cols = [c for c in feat_cols if c not in num_cols and c not in bool_cols]
print('num', len(num_cols), 'bool', len(bool_cols), 'cat', cat_cols)

corr = m[num_cols + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
corr = corr.reindex(corr.abs().sort_values(ascending=False).index)
print('\ntop |corr| with target:')
print(corr.head(30).round(3))