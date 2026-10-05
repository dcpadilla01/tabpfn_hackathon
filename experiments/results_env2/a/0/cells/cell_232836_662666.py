
import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
f4 = agent_api.load_saved('feats_v4.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')

FEATS3 = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]
FEATS4 = [c for c in f4.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4,
              min_child_weight=20, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              seed=7, eval_metric='mae')

def fit_eval(df, feats, snaps_train, snap_val):
    tr = df[df.snapshot_day.isin(snaps_train)]
    va = df[df.snapshot_day==snap_val]
    dtr = xgb.DMatrix(tr[feats], label=tr.future_spend_4w)
    dva = xgb.DMatrix(va[feats])
    bst = xgb.train(params, dtr, num_boost_round=600, verbose_eval=False)
    return mean_absolute_error(va.future_spend_4w, bst.predict(dva))

for snap_val in [347, 375, 403, 431]:
    tr_snaps = [s for s in [95,123,151,179,207,235,263,291,319] if s < snap_val]
    mae3 = fit_eval(f3, FEATS3, tr_snaps, snap_val)
    mae4 = fit_eval(f4, FEATS4, tr_snaps, snap_val)
    print(f"val {snap_val}: v3 {mae3:.3f}  v4 {mae4:.3f}")
