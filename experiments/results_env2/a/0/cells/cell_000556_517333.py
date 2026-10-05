import pandas as pd, numpy as np, agent_api, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor
feats = agent_api.load_saved('feats_v4.parquet')
tt = agent_api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='left')
for c in feats.columns:
    if feats[c].dtype.kind not in 'fi':
        df[c] = df[c].astype('category').cat.codes
feat_cols = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = df[feat_cols].astype(float).values
y = df['future_spend_4w'].values
tr = np.where(df.snapshot_day <= 403)[0]; te = np.where(df.snapshot_day == 431)[0]
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
def mae(p): return round(np.abs(p-yte).mean(),3)
def xq(Xtr,ytr,Xte,alpha=0.5,seed=7):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=5,
        min_child_weight=40, learning_rate=0.08, n_estimators=400, subsample=0.9,
        colsample_bytree=0.8, n_jobs=8, tree_method='hist', random_state=seed)
    m.fit(Xtr,ytr); return m.predict(Xte)

# H: HistGBR quantile
hg = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=300,
    learning_rate=0.08, max_depth=None, max_leaf_nodes=31, min_samples_leaf=60,
    l2_regularization=1.0, random_state=7).fit(Xtr, ytr)
ph = hg.predict(Xte); print('H HistGBR q0.5:', mae(ph))
# I: avg xgb(seeds) + HistGBR
pb = np.mean([xq(Xtr,ytr,Xte,seed=s) for s in (1,2,3,4,5)], axis=0)
print('I avg(xgb-bag, histgbr):', mae((pb+ph)/2), '| w 0.7/0.3:', mae(0.7*pb+0.3*ph))
# J: train only on later snapshots (day>=235)
late = np.where(df.snapshot_day<=403)[0][df.snapshot_day.values[tr]>=235]
pl2 = xq(X[late], y[late], Xte); print('J late-only:', mae(pl2))
# K: drop demographic cols
demo_idx = [i for i,c in enumerate(feat_cols) if c.startswith(('classification','homeowner','kid','has_demo'))]
Xk = np.delete(X, demo_idx, axis=1)
pk = xq(Xk[tr], ytr, Xk[te]); print('K no-demo:', mae(pk))
# L: drop snap cols
snap_idx = [i for i,c in enumerate(feat_cols) if c.startswith('snap')]
Xl = np.delete(X, snap_idx, axis=1)
pl3 = xq(Xl[tr], ytr, Xl[te]); print('L no-snap:', mae(pl3))
# M: bag with raw+log mix + histgbr
plog = np.expm1(xq(Xtr, np.log1p(ytr), Xte))
print('M best-mix avg(pb,pl,ph):', mae((pb+pl+ph)/3))
