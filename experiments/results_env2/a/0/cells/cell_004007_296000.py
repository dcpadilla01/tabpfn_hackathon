import agent_api as A, pandas as pd, numpy as np, time
from xgboost import XGBRegressor

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
feat_cols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = feats[feat_cols].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        codes = pd.factorize(X[c])[0].astype(float); codes[codes==-1]=np.nan; X[c]=codes
Xi = X.values
y_full = feats[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='left')['future_spend_4w'].values
day = feats.snapshot_day.values
val_mask = day>=459
train_mask = (day<=431) & ~np.isnan(y_full)
Xtr, ytr = Xi[train_mask], y_full[train_mask]
Xval = Xi[val_mask]
print('train rows:', len(ytr), 'val rows:', int(val_mask.sum()))

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
t0=time.time()
preds=[]
for d,mcw in configs:
    for s in [0,1]:
        m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
                         subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                         n_jobs=4, random_state=s, tree_method='hist')
        m.fit(Xtr, ytr)
        preds.append(m.predict(Xval))
pred = np.clip(np.mean(np.array(preds),0), 0, None)
out = feats.loc[val_mask, ['household_key','snapshot_day']].copy()
out['prediction'] = pred
print('out', out.shape, 'nan:', int(out.prediction.isna().sum()), 'finite:', bool(np.isfinite(out.prediction).all()))
p = A.save_table(out, 'pred_e014.parquet')
print('saved:', p, 'time', round(time.time()-t0,1))
