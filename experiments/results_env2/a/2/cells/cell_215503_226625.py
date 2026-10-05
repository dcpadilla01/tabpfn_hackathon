import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()]
va = df[df.snapshot_day>=459]
feats_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
Xtr, ytr = tr[feats_cols].to_numpy(np.float32), tr.future_spend_4w.to_numpy(np.float32)
Xva = va[feats_cols].to_numpy(np.float32)
print('train', Xtr.shape, 'val', Xva.shape)

model = xgb.XGBRegressor(
    n_estimators=1600, learning_rate=0.03, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0.5,
    objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
model.fit(Xtr, ytr, verbose=False)
pv = model.predict(Xva)
imp = pd.Series(model.feature_importances_, index=feats_cols).sort_values(ascending=False)
print(imp.head(20).round(4).to_string())

pred = va[['household_key','snapshot_day']].copy()
pred['prediction'] = pv
p = api.save_table(pred, 'pred_e003.parquet')
print('saved', p, len(pred))
