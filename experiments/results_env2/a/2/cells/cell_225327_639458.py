import pandas as pd, numpy as np, xgboost as xgb
from agent_api import load_saved, train_targets, KEYS, TARGET

feats = load_saved('feats_v3.parquet')
tt = train_targets()
df = tt.merge(feats, on=KEYS, how='left')
obj_cols = [c for c in df.columns if df[c].dtype == object]
for c in obj_cols:
    df[c] = df[c].astype('category').cat.codes
FEATS = [c for c in df.columns if c not in KEYS+[TARGET]]
df[FEATS] = df[FEATS].fillna(-1)

tr = df[df.snapshot_day <= 403]
va = df[df.snapshot_day == 431]

def fit_med(seed, n=1200, lr=0.03, md=7, mcw=10):
    m = xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=md, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror',
                         quantile_alpha=0.5, random_state=seed, n_jobs=8, tree_method='hist')
    m.fit(tr[FEATS], tr[TARGET])
    return m

preds = []
for s in [1,2,3]:
    m = fit_med(s)
    preds.append(m.predict(va[FEATS]))
p = np.mean(preds, axis=0)
mae = np.abs(p - va[TARGET]).mean()
print('E005-like MAE on day 431:', round(mae,3))

err = p - va[TARGET].values
ae = np.abs(err)
va2 = va.copy()
va2['pred'] = p; va2['err'] = err; va2['ae'] = ae
va2['tq'] = pd.qcut(va2[TARGET], 10, duplicates='drop')
g = va2.groupby('tq', observed=True).agg(n=('ae','size'), y_mean=(TARGET,'mean'), p_mean=('pred','mean'), bias=('err','mean'), mae=('ae','mean'))
print(g.round(1))
# zero spenders
z = va2[va2[TARGET]==0]
print('zero-spend rows:', len(z), 'mean pred on them:', round(z['pred'].mean(),1), 'share of total MAE:', round(z['ae'].sum()/ae.sum(),3))
nz = va2[va2[TARGET]>0]
print('nonzero MAE:', round(nz['ae'].mean(),2))
# top decile contribution
top = va2[va2[TARGET] >= va2[TARGET].quantile(0.9)]
print('top-decile rows:', len(top), 'share of MAE:', round(top['ae'].sum()/ae.sum(),3), 'their MAE:', round(top['ae'].mean(),1))
# feature importance
m0 = fit_med(1)
imp = pd.Series(m0.feature_importances_, index=FEATS).sort_values(ascending=False)
print(imp.head(25).round(4))
