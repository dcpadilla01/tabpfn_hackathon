
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

feats = api.load_saved('feats_v3.parquet')
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
def mae(p): return round(np.abs(np.clip(p,0,None)-yho).mean(),3)

base = dict(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)
t0=time.time()
mx = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr,ytr)
p_xgb = mx.predict(Xho); print('xgb med', mae(p_xgb), round(time.time()-t0,1),'s')

t0=time.time()
h1 = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=600, learning_rate=0.05,
     max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h1 = h1.predict(Xho); print('histgb quantile', mae(p_h1), round(time.time()-t0,1),'s')
t0=time.time()
h2 = HistGradientBoostingRegressor(loss='absolute_error', max_iter=600, learning_rate=0.05,
     max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=7).fit(Xtr,ytr)
p_h2 = h2.predict(Xho); print('histgb abserr', mae(p_h2), round(time.time()-t0,1),'s')

for w in [0.2,0.3,0.4,0.5]:
    print(f'blend xgb+{w}*hist_q MAE', mae((1-w)*p_xgb + w*p_h1))
for w in [0.2,0.3,0.4]:
    print(f'blend xgb+{w}*(h1+h2)/2 MAE', mae((1-w)*p_xgb + w*0.5*(p_h1+p_h2)))

# feature selection by gain importance
imp = mx.feature_importances_
order = np.argsort(-imp)
for K in [30, 50, 70]:
    cols = [feat_cols[i] for i in order[:K]]
    m = xgb.XGBRegressor(**base, objective='reg:quantileerror', quantile_alpha=0.5).fit(Xtr[:,order[:K]],ytr)
    print(f'top{K} feat MAE', mae(m.predict(Xho[:,order[:K]])))
print('top10:', [feat_cols[i] for i in order[:10]])
