
import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
BASE = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

w = f3[['spend_28','spend28_lag1','spend28_lag2','spend28_lag3']]
f3['med4w'] = w.median(axis=1)
f3['min4w'] = w.min(axis=1)
f3['max4w'] = w.max(axis=1)
f3['zero_weeks8'] = (f3[['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7']]==0).sum(axis=1)
f3['spend7_over_28'] = f3.spend_7/(f3.spend_28+1)
f3['dsl_vs_gap'] = f3.days_since_last/(f3.gap_mean_84+1)
FEATS = BASE + ['med4w','min4w','max4w','zero_weeks8','spend7_over_28','dsl_vs_gap']

def run(snap_val, mode, blend_w, feats, seed=7, rounds=600, lr=0.05):
    tr = f3[f3.snapshot_day.isin([s for s in [95,123,151,179,207,235,263,291,319] if s<snap_val])]
    va = f3[f3.snapshot_day==snap_val]
    if mode=='log':
        y = np.log1p(tr.future_spend_4w)
        p = dict(objective='reg:squarederror', max_depth=4, min_child_weight=20,
                 learning_rate=lr, subsample=0.8, colsample_bytree=0.8, seed=seed)
    else:
        y = tr.future_spend_4w
        p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
                 learning_rate=lr, subsample=0.8, colsample_bytree=0.8, seed=seed)
    bst = xgb.train(p, xgb.DMatrix(tr[feats], label=y), num_boost_round=rounds, verbose_eval=False)
    pr = bst.predict(xgb.DMatrix(va[feats]))
    pr = np.expm1(pr) if mode=='log' else pr
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend)

vals = [347,375,403,431]
for name, mode, feats in [('quant_base','q',BASE), ('quant+med4w','q',FEATS),
                          ('log_base','log',BASE), ('log+med4w','log',FEATS)]:
    maes = [run(v, mode, 0.7, feats) for v in vals]
    print(f"{name:14s} local MAE {np.mean(maes):.3f}  ({' '.join(f'{m:.1f}' for m in maes)})")
