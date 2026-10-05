import pandas as pd, numpy as np, xgboost as xgb, time
from sklearn.metrics import roc_auc_score
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

def fit_med(seed, X, y, qa=0.5, n=1200):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=0.03, max_depth=7, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=qa,
        random_state=seed, n_jobs=8, tree_method='hist').fit(X, y)

t0=time.time()
pa = np.mean([fit_med(s, tr[FEATS], tr[TARGET]).predict(va[FEATS]) for s in [1,2]], axis=0)
print('base MAE 431:', round(np.abs(pa-yv).mean(),3), 'time', round(time.time()-t0,1))

# 1) median calibration: fraction y < pred by pred decile
q = pd.qcut(pa, 10, duplicates='drop')
frac = pd.DataFrame({'q':q, 'y':yv, 'p':pa}).groupby('q', observed=True).apply(lambda g: (g.y<g.p).mean(), include_groups=False)
print('frac(y<pred) by pred decile (0.5 = perfect median):')
print(frac.round(2))

# 2) zero classifier
yz_tr = (tr[TARGET]==0).astype(int)
clf = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, random_state=1, n_jobs=8, tree_method='hist', eval_metric='logloss')
clf.fit(tr[FEATS], yz_tr)
p0 = clf.predict_proba(va[FEATS])[:,1]
yz_va = (va[TARGET]==0).astype(int)
print('zero-clf AUC 431:', round(roc_auc_score(yz_va, p0),3), 'base rate:', round(yz_va.mean(),3))
for thr in [0.3,0.4,0.5,0.6]:
    pf = np.where(p0>thr, 0.0, pa)
    print(f'override thr={thr}: MAE', round(np.abs(pf-yv).mean(),3), 'zeroed rows:', int((p0>thr).sum()))

# 3) quantile_alpha sweep (single seed, cheap-ish)
for qa in [0.45, 0.55]:
    p = fit_med(1, tr[FEATS], tr[TARGET], qa=qa).predict(va[FEATS])
    print(f'qa={qa}: MAE', round(np.abs(p-yv).mean(),3))
