
import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
BASE = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

def run(snap_val, seeds, rounds=600, lr=0.05, md=4, mcw=20, ss=0.8, cs=0.8, blend_w=0.85):
    tr = f3[f3.snapshot_day.isin([s for s in [95,123,151,179,207,235,263,291,319] if s<snap_val])]
    va = f3[f3.snapshot_day==snap_val]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md, min_child_weight=mcw,
             learning_rate=lr, subsample=ss, colsample_bytree=cs)
    preds = []
    for s in seeds:
        p['seed']=s
        bst = xgb.train(p, xgb.DMatrix(tr[BASE], label=tr.future_spend_4w), num_boost_round=rounds, verbose_eval=False)
        preds.append(bst.predict(xgb.DMatrix(va[BASE])))
    pr = np.mean(preds, axis=0)
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend)

vals = [347,375,403,431]
for md, mcw in [(4,20),(5,20),(4,10),(6,30),(5,40)]:
    maes = [run(v, [7], md=md, mcw=mcw) for v in vals]
    print(f"md={md} mcw={mcw}: local MAE {np.mean(maes):.3f}")
print()
for lr, rounds in [(0.03,1000),(0.05,600),(0.08,400)]:
    maes = [run(v, [7], lr=lr, rounds=rounds) for v in vals]
    print(f"lr={lr} rounds={rounds}: local MAE {np.mean(maes):.3f}")
