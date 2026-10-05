import agent_api, pandas as pd, numpy as np
df = agent_api.load_saved('e015_stack.parquet')
tt = agent_api.train_targets()
d = df.merge(tt, on=['household_key','snapshot_day'], how='left').reset_index(drop=True)
is_tr = d['future_spend_4w'].notna().values
cat_cols = [c for c in df.columns if str(df[c].dtype)=='category']
base_drop = set(['household_key','snapshot_day','stack_ridge_log','stack_ridge'])|set(cat_cols)
num_cols = [c for c in df.columns if c not in base_drop]
X = np.hstack([d[num_cols].astype(float).values, pd.get_dummies(d[cat_cols].astype(str), dummy_na=True).values.astype(np.float64)])
print('inf count:', np.isinf(X).sum())
infcols = np.isinf(X).sum(axis=0)
bad = [num_cols[i] if i<len(num_cols) else ('DUM%d'%i) for i in np.where(infcols>0)[0]]
print('cols with inf:', bad[:20])
print('rows with inf:', np.isinf(X).any(axis=1).sum())
days = d['snapshot_day'].values
rows_inf = np.where(np.isinf(X).any(axis=1))[0]
print('inf rows by day:', pd.Series(days[rows_inf]).value_counts().sort_index().to_dict())
# check per-column inf by day
for c in bad[:5]:
    j = num_cols.index(c) if c in num_cols else None
    print(c, 'inf by day:', pd.Series(days[np.isinf(X[:,j])]).value_counts().to_dict())
