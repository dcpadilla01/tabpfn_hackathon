
import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
f4 = agent_api.load_saved('feats_v4.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
F3 = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]
F4 = [c for c in f4.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

def run(df, feats, snap_val, seeds, rounds=400, lr=0.08, md=5, mcw=40, blend_w=0.85):
    tr = df[df.snapshot_day.isin([s for s in [95,123,151,179,207,235,263,291,319] if s<snap_val])]
    va = df[df.snapshot_day==snap_val]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md, min_child_weight=mcw,
             learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
    preds = []
    for s in seeds:
        p['seed']=s
        bst = xgb.train(p, xgb.DMatrix(tr[feats], label=tr.future_spend_4w), num_boost_round=rounds, verbose_eval=False)
        preds.append(bst.predict(xgb.DMatrix(va[feats])))
    pr = np.mean(preds, axis=0)
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend)

vals = [347,375,403,431]
for name, df, feats in [('v3', f3, F3), ('v4', f4, F4)]:
    for seeds in [[7],[7,8,9,10,11,12,13,14]]:
        maes = [run(df, feats, v, seeds) for v in vals]
        print(f"{name} seeds={len(seeds)}: local MAE {np.mean(maes):.3f}  ({' '.join(f'{m:.1f}' for m in maes)})")
