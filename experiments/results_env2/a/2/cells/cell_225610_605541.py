import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.isotonic import IsotonicRegression
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
TRAIN_DAYS = sorted(tt.snapshot_day.unique())

def fit_med(seed, X, y):
    return xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        random_state=seed, n_jobs=8, tree_method='hist').fit(X, y)

t0=time.time()
oof = []
for d in TRAIN_DAYS:
    trd = df[df.snapshot_day != d]
    vad = df[df.snapshot_day == d]
    p = np.mean([fit_med(s, trd[FEATS], trd[TARGET]).predict(vad[FEATS]) for s in [1,2]], axis=0)
    oof.append(pd.DataFrame({'household_key': vad.household_key.values, 'snapshot_day': d,
                             'pred': p, 'y': vad[TARGET].values}))
oof = pd.concat(oof, ignore_index=True)
print('OOF built in', round(time.time()-t0,1), 's; OOF MAE:', round(np.abs(oof.pred-oof.y).mean(),3))
oof['tq'] = pd.qcut(oof.y, 10, duplicates='drop')
oof['bias'] = oof.pred - oof.y
print(oof.groupby('tq', observed=True)[['y','pred','bias']].mean().round(1))
iso = IsotonicRegression(y_min=0, out_of_bounds='clip'); iso.fit(oof.pred.values, oof.y.values)
oof['pc'] = iso.predict(oof.pred.values)
print('OOF MAE after pooled recal:', round(np.abs(oof.pc-oof.y).mean(),3))
oof['tq2'] = pd.qcut(oof.pred, 10, duplicates='drop')
print(oof.groupby('tq2', observed=True)[['y','pred']].mean().round(1))
