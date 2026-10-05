import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb

tt = A.train_targets()
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g.round(1))
print('overall mean/median/std:', round(tt.future_spend_4w.mean(),2), tt.future_spend_4w.median(), round(tt.future_spend_4w.std(),2))
print('zero frac:', round((tt.future_spend_4w==0).mean(),4))

oof = A.load_saved('oof_e008.parquet')
print('oof cols:', oof.columns.tolist()); print(oof.head(3))

f3 = A.load_saved('feats_v3.parquet')
print('feats_v3 shape:', f3.shape)
tr = f3.merge(tt, on=['household_key','snapshot_day'])
feats = [c for c in f3.columns if c not in ('household_key','snapshot_day')]
X = tr[feats].astype(float); X = X.fillna(X.median()); y = tr.future_spend_4w
t0=time.time()
m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=7, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8)
m.fit(X, y)
imp = pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False)
print(imp.head(30).round(4).to_string())
print('fit time', round(time.time()-t0,1))
