
import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f4 = agent_api.load_saved('feats_v4.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
F4 = [c for c in f4.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

def run(snap_val, seeds, blend_w, rounds=400, lr=0.08, md=5, mcw=40):
    tr = f4[f4.snapshot_day.isin([s for s in [95,123,151,179,207,235,263,291,319] if s<snap_val])]
    va = f4[f4.snapshot_day==snap_val]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md, min_child_weight=mcw,
             learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
    preds = []
    for s in seeds:
        p['seed']=s
        bst = xgb.train(p, xgb.DMatrix(tr[F4], label=tr.future_spend_4w), num_boost_round=rounds, verbose_eval=False)
        preds.append(bst.predict(xgb.DMatrix(va[F4])))
    pr = np.mean(preds, axis=0)
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend)

vals = [347,375,403,431]
for bw in [0.8,0.85,0.9,0.95]:
    maes = [run(v, [7,8,9,10,11,12,13,14], bw) for v in vals]
    print(f"blend_w {bw}: local MAE {np.mean(maes):.3f}")
