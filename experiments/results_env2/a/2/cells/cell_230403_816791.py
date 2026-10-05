import pandas as pd, numpy as np, xgboost as xgb, time
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
for c in df.columns:
    if df[c].dtype == object: df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)
tr = df[df.snapshot_day <= 403]; va = df[df.snapshot_day == 431]
yv = va[TARGET].values

def base_params(obj_kwargs, seed):
    d = dict(n_estimators=1200, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=8, tree_method='hist')
    d.update(obj_kwargs); return d

t0=time.time()
# A: median ensemble (E005-like, 2 seeds)
pA = np.mean([xgb.XGBRegressor(**base_params(dict(objective='reg:quantileerror', quantile_alpha=0.5), s)
              ).fit(tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
# B: log-space L2
mB = xgb.XGBRegressor(**base_params(dict(objective='reg:squarederror'), 1))
mB.fit(tr[FEATS], np.log1p(tr[TARGET]))
pB = np.expm1(mB.predict(va[FEATS]))
# C: absoluteerror objective
pC = np.mean([xgb.XGBRegressor(**base_params(dict(objective='reg:absoluteerror'), s)
              ).fit(tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
print('time', round(time.time()-t0,1))
print('A median-ens :', round(np.abs(pA-yv).mean(),3))
print('B log-L2     :', round(np.abs(pB-yv).mean(),3))
print('C abserror   :', round(np.abs(pC-yv).mean(),3))
for w in [0.3,0.5,0.7]:
    print(f'blend A+B w={w}:', round(np.abs(w*pA+(1-w)*pB-yv).mean(),3))
print('A+B+C equal:', round(np.abs((pA+pB+pC)/3-yv).mean(),3))
# bias by decile for B
q = pd.qcut(yv, 8, duplicates='drop')
tmp = pd.DataFrame({'q':q,'y':yv,'pA':pA,'pB':pB})
print(tmp.groupby('q', observed=True)[['y','pA','pB']].mean().round(0))
