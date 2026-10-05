import pandas as pd, numpy as np, xgboost as xgb, warnings, time
from sklearn.ensemble import HistGradientBoostingRegressor
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
m = tt.merge(f3, on=['household_key','snapshot_day'], how='left')
DROP = ['index','household_key','snapshot_day','future_spend_4w']
FEATS = [c for c in m.columns if c not in DROP]
m[FEATS] = m[FEATS].fillna(-1)
for c in ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner','kids']:
    m[c] = m[c].astype('category').cat.codes
tr = m[m.snapshot_day <= 347]; te = m[m.snapshot_day.isin([375,431])]
def xgbfit(df_tr, y, params=None):
    p = dict(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=4, min_child_weight=20,
             learning_rate=0.05, n_estimators=800, subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=4)
    p.update(params or {})
    mod = xgb.XGBRegressor(**p); mod.fit(df_tr[FEATS], y); return mod
def mae(pred):
    return {s: round(float(np.abs(pred[te.snapshot_day==s]-te[te.snapshot_day==s].future_spend_4w.values).mean()),3) for s in [375,431]}, round(float(np.abs(pred-te.future_spend_4w.values).mean()),3)
# A: pure xgb quantile
pA = np.clip(xgbfit(tr, tr.future_spend_4w).predict(te[FEATS]),0,None)
print('xgb quantile      ', mae(pA))
# B: xgb quantile on log1p target
modB = xgbfit(tr, np.log1p(tr.future_spend_4w))
pB = np.clip(np.expm1(modB.predict(te[FEATS])),0,None)
print('xgb quantile log  ', mae(pB))
# C: sklearn HGB absolute_error
modC = HistGradientBoostingRegressor(loss='absolute_error', max_iter=500, learning_rate=0.05, max_depth=None, min_samples_leaf=40, l2_regularization=1.0)
modC.fit(tr[FEATS], tr.future_spend_4w)
pC = np.clip(modC.predict(te[FEATS]),0,None)
print('sklearn HGB abs   ', mae(pC))
# D: ensemble xgb + HGB
print('ens xgb+HGB 50/50 ', mae(0.5*pA+0.5*pC))
print('ens xgb+HGB 70/30 ', mae(0.7*pA+0.3*pC))
print('ens xgb+HGBlog 50 ', mae(0.5*pA+0.5*pB))
print('ens all 3         ', mae((pA+pB+pC)/3))
# E: shrinkage estimator: household median of past cycle spends (lag1-3) shrunk to global
cyc = te[['lag1_spend','lag2_spend','lag3_spend']].values
n = (~np.isnan(cyc)).sum(1) if np.isnan(cyc).any() else (cyc>0).sum(1)
med = np.nanmedian(np.where(cyc>0, cyc, np.nan), axis=1)
gm = tr.future_spend_4w.median()
pE = np.clip((np.nan_to_num(med)*n + gm*3)/(n+3), 0, None)
print('shrinkage est only', mae(pE))
