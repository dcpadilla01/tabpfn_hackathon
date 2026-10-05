
import agent_api as api, pandas as pd, numpy as np, xgboost as xgb

feats = api.load_saved("feats_v1.parquet")
tt = api.train_targets()
tr = feats[feats.snapshot_day.isin(api.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
va = feats[feats.snapshot_day.isin(api.snapshot_days()['validation'])]
drop = ['household_key','snapshot_day','future_spend_4w']
Xtr, ytr = tr.drop(columns=drop), tr.future_spend_4w
Xva = va.drop(columns=drop)

model = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, tree_method='hist',
                         n_jobs=8, random_state=0)
model.fit(Xtr, ytr)
pred = model.predict(Xva)
sub = va[['household_key','snapshot_day']].copy()
sub['prediction'] = pred
print(sub.prediction.describe())
path = api.save_table(sub, "pred_e001")
print(path)
