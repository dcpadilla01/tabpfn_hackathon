import agent_api, pandas as pd, numpy as np, xgboost as xgb

f = agent_api.load_saved('feats_v4.parquet')
t = agent_api.train_targets()
print('feats rows', len(f), 'targets rows', len(t))
print(f.groupby('snapshot_day').size())

d = f.merge(t, on=['household_key','snapshot_day'], how='left')
train_days = agent_api.snapshot_days()['train']
val_days = agent_api.snapshot_days()['validation']
tr = d[d.snapshot_day.isin(train_days)].copy()
va = d[d.snapshot_day.isin(val_days)].copy()
print('train rows', len(tr), 'val rows', len(va))

drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in d.columns if c not in drop]
Xtr, ytr = tr[feat_cols], tr['future_spend_4w']
Xva = va[feat_cols]

model = xgb.XGBRegressor(
    objective='reg:quantileerror', quantile_alpha=0.5,
    n_estimators=700, learning_rate=0.05, max_depth=4, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8, random_state=0)
model.fit(Xtr, ytr)
pred = 0.7*np.maximum(model.predict(Xva),0) + 0.3*np.maximum(va['exp4w_blend'].values,0)

out = pd.DataFrame({'household_key': va['household_key'].values,
                    'snapshot_day': va['snapshot_day'].values,
                    'prediction': pred})
path = agent_api.save_table(out, 'pred_e008.parquet')
print(path, len(out))
imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(15))
