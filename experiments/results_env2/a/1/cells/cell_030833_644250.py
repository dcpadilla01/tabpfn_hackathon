import agent_api as api
import pandas as pd, numpy as np, xgboost as xgb, time
allF = api.load_saved('allF.parquet')
feat_cols = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X = allF[feat_cols].astype(float); y = allF['future_spend_4w'].astype(float); days = allF['snapshot_day'].astype(int)
vmask = days>=459
Xv = X[vmask]; val = allF[vmask]
tr = (days>=151)&(days<=403)&y.notna()
w = 0.5**((403-days[tr].values)/140)
preds=[]; t0=time.time()
for seed in [1,7,42]:
    m = xgb.XGBRegressor(n_estimators=2400, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', max_depth=6, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        nthread=-1, seed=seed)
    m.fit(X[tr], y[tr], sample_weight=w)
    preds.append(m.predict(Xv))
    print("seed", seed, "done %.0fs" % (time.time()-t0), flush=True)
p = np.mean(preds, axis=0)
base = 0.31*val['spend_84'].values
b = 0.5*p + 0.5*base
b = np.where(b<10, 0, b)
out = pd.DataFrame({'household_key': val['household_key'].values,
                    'snapshot_day': val['snapshot_day'].values,
                    'prediction': b})
print(out.shape)
print(out.groupby('snapshot_day')['prediction'].agg(['count','mean','median']).round(2))
print("pred mean overall %.2f (E005 was 127.10)" % out.prediction.mean())
path = api.save_table(out, 'e014_preds.parquet')
print("saved:", path)
