
import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved('feats_v3.parquet')
print("feats_v3 shape:", f3.shape)
print("cols:", list(f3.columns))
print(f3.head(3))

tt = agent_api.train_targets()
print("\ntrain_targets shape:", tt.shape)
print(tt.head(3))
print(tt.future_spend_4w.describe())

p7 = agent_api.load_saved('pred_e007.parquet')
print("\npred_e007 shape:", p7.shape, "cols:", list(p7.columns))
print(p7.head(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

f3 = agent_api.load_saved('feats_v3.parquet')
print(f3.dtypes.to_string())
tt = agent_api.train_targets()
m = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("\nmerged:", m.shape)
# local split: train snaps 151..403, local val 431
print(sorted(m.snapshot_day.unique()))
print("rows per snap:\n", m.groupby('snapshot_day').size())
print("\nmean target by snap:\n", m.groupby('snapshot_day').future_spend_4w.agg(['mean','median']))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
f4 = agent_api.load_saved('feats_v4.parquet')
print("feats_v4 shape:", f4.shape)
print([c for c in f4.columns if c not in agent_api.load_saved('feats_v3.parquet').columns])


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

f3 = agent_api.load_saved('feats_v3.parquet').copy()
f4 = agent_api.load_saved('feats_v4.parquet').copy()
tt = agent_api.train_targets()
f4 = f4.merge(tt, on=['household_key','snapshot_day'], how='left')
f4['future_spend_4w'] = f4['future_spend_4w'].fillna(-1)

FEATS3 = [c for c in f3.columns if c not in ('index','household_key','snapshot_day')]
FEATS4 = [c for c in f4.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

def fit_eval(df, feats, snaps_train, snap_val, params, seed=7):
    tr = df[df.snapshot_day.isin(snaps_train)]
    va = df[df.snapshot_day==snap_val]
    dtr = xgb.DMatrix(tr[feats], label=tr.future_spend_4w)
    dva = xgb.DMatrix(va[feats])
    bst = xgb.train(params, dtr, num_boost_round=600, evals=[(dtr,'tr')], verbose_eval=False)
    return mean_absolute_error(va.future_spend_4w, bst.predict(dva)), bst

params = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4,
              min_child_weight=20, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
              seed=7, eval_metric='mae')

for snap_val in [347, 375, 403, 431]:
    tr_snaps = [s for s in [95,123,151,179,207,235,263,291,319] if s < snap_val]
    mae3,_ = fit_eval(f3, FEATS3, tr_snaps, snap_val, params)
    mae4,_ = fit_eval(f4, FEATS4, tr_snaps, snap_val, params)
    print(f"val {snap_val}: v3 {mae3:.3f}  v4 {mae4:.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np
f4 = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
print([c for c in f4.columns if 'future' in c or 'spend_4w' in c])
m = f4.merge(tt, on=['household_key','snapshot_day'], how='left')
print(m.shape)
print([c for c in m.columns if 'future' in c])
print(m.future_spend_4w.isna().sum())


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')

# extra robust-level features
w = f3[['spend_28','spend28_lag1','spend28_lag2','spend28_lag3']]
f3['med4w'] = w.median(axis=1)
f3['min4w'] = w.min(axis=1)
f3['max4w'] = w.max(axis=1)
f3['zero_weeks8'] = (f3[['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7']]==0).sum(axis=1)
f3['spend7_over_28'] = f3.spend_7/(f3.spend_28+1)
f3['dsl_vs_gap'] = f3.days_since_last/(f3.gap_mean_84+1)

BASE = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]
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
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend), pr

vals = [347,375,403,431]
for name, mode, feats in [('quant_base','q',BASE), ('quant+med4w','q',FEATS),
                          ('log_base','log',BASE), ('log+med4w','log',FEATS)]:
    maes = [run(v, mode, 0.7, feats)[0] for v in vals]
    print(f"{name:14s} local MAE {np.mean(maes):.3f}  ({' '.join(f'{m:.1f}' for m in maes)})")


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

tt = agent_api.train_targets()
f3 = agent_api.load_saved('feats_v3.parquet').merge(tt, on=['household_key','snapshot_day'], how='left')
BASE = [c for c in f3.columns if c not in ('index','household_key','snapshot_day','future_spend_4w')]

def run(snap_val, blend_w, seeds, rounds=600, lr=0.05, md=4, mcw=20):
    tr = f3[f3.snapshot_day.isin([s for s in [95,123,151,179,207,235,263,291,319] if s<snap_val])]
    va = f3[f3.snapshot_day==snap_val]
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md, min_child_weight=mcw,
             learning_rate=lr, subsample=0.8, colsample_bytree=0.8)
    preds = []
    for s in seeds:
        p['seed']=s
        bst = xgb.train(p, xgb.DMatrix(tr[BASE], label=tr.future_spend_4w), num_boost_round=rounds, verbose_eval=False)
        preds.append(bst.predict(xgb.DMatrix(va[BASE])))
    pr = np.mean(preds, axis=0)
    return mean_absolute_error(va.future_spend_4w, blend_w*pr + (1-blend_w)*va.exp4w_blend)

vals = [347,375,403,431]
for bw in [0.5,0.6,0.7,0.8,0.9,1.0]:
    maes = [run(v, bw, [7]) for v in vals]
    print(f"blend_w {bw}: local MAE {np.mean(maes):.3f}")
print()
for seeds in [[7],[7,8,9,10],[7,8,9,10,11,12,13,14]]:
    maes = [run(v, 0.7, seeds) for v in vals]
    print(f"seeds {seeds}: local MAE {np.mean(maes):.3f}")


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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


# ---- cell ----

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
