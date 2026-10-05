import numpy as np, pandas as pd, xgboost as xgb
F = agent_api.load_saved('feats_v1.parquet')
tt = agent_api.train_targets()
tr = F[F.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])]
va = F[F.snapshot_day.isin([459,487,515,543])]
m = tr.merge(tt, on=['household_key','snapshot_day'])
print(m.shape)
feat_cols = [c for c in F.columns if c not in ('household_key','snapshot_day')]
X, y = m[feat_cols].values, m.future_spend_4w.values
Xv = va[feat_cols].values
mdl = xgb.XGBRegressor(n_estimators=900, learning_rate=0.05, max_depth=7, subsample=0.8,
                       colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                       objective='reg:absoluteerror', tree_method='hist', n_jobs=8, random_state=0)
mdl.fit(X, y)
pred = mdl.predict(Xv)
out = va[['household_key','snapshot_day']].copy()
out['prediction'] = np.clip(pred, 0, None)
p = agent_api.save_table(out, 'pred_e001')
imp = pd.Series(mdl.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(imp.head(12))
