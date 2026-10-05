import agent_api, numpy as np, pandas as pd, re
t = agent_api.load_saved('e011_table.parquet')
print('e011 shape', t.shape)
cols = list(t.columns)
print('n cols', len(cols))
groups = {}
for c in cols:
    pre = re.split(r'[_0-9]', c)[0]
    groups.setdefault(pre, []).append(c)
for k in sorted(groups):
    print(k, len(groups[k]), groups[k][:10])
tt = agent_api.train_targets()
print('targets', tt.shape)
m = tt.merge(t, on=['household_key','snapshot_day'], how='left')
print('merged', m.shape, 'rows with NA:', int(m.isna().any(axis=1).sum()))
y = m['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2))
print('zero share', round(float((y==0).mean()),4))
feat_cols = [c for c in cols if c not in ('household_key','snapshot_day')]
num_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(m[c])]
print('numeric', len(num_cols), 'nonnumeric', [c for c in feat_cols if c not in num_cols][:20])
cor = m[num_cols].apply(lambda s: s.corr(y)).dropna()
order = cor.abs().sort_values(ascending=False).index
print('--- top corr with target ---')
print(cor[order[:35]].round(3))
print('--- weakest ---')
print(cor[order[-12:]].round(3))
print(m.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1))
