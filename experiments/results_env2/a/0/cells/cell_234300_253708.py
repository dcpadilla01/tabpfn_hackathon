
import agent_api, pandas as pd, numpy as np, xgboost as xgb

tt = agent_api.train_targets()
f4 = agent_api.load_saved('feats_v4.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
F4 = [c for c in f4.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

TRAIN_SNAPS = [95,123,151,179,207,235,263,291,319,347,375,403,431]
VAL_SNAPS = [459,487,515,543]

p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=5, min_child_weight=40,
         learning_rate=0.08, subsample=0.8, colsample_bytree=0.8)
preds = []
for s in [7,8,9,10,11,12,13,14]:
    p['seed']=s
    bst = xgb.train(p, xgb.DMatrix(f4[f4.snapshot_day.isin(TRAIN_SNAPS)][F4],
                                   label=f4[f4.snapshot_day.isin(TRAIN_SNAPS)].future_spend_4w),
                    num_boost_round=400, verbose_eval=False)
    va = f4[f4.snapshot_day.isin(VAL_SNAPS)]
    preds.append(bst.predict(xgb.DMatrix(va[F4])))
pr = np.mean(preds, axis=0)

va = f4[f4.snapshot_day.isin(VAL_SNAPS)].copy()
va['prediction'] = 0.9*pr + 0.1*va.exp4w_blend
out = va[['household_key','snapshot_day','prediction']].reset_index(drop=True)
print(out.shape, out.snapshot_day.value_counts().to_dict())
path = agent_api.save_table(out, 'pred_e011.parquet')
print(path)
