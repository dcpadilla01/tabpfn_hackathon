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
etr = day==431; ye = y_full[etr]
blend_feat = np.nan_to_num(feats['exp4w_blend'].values)

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
def fit(Xtr, ytr, depth, mcw, seed):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                     n_jobs=4, random_state=seed, tree_method='hist')
    m.fit(Xtr, ytr); return m

t0=time.time()
# local check on 431: train <=403, 8 configs x 2 seeds
mtr = (day<=403) & ~np.isnan(y_full)
preds=[]
for d,mcw in configs:
    for s in [0,1]:
        preds.append(fit(Xi[mtr], y_full[mtr], d, mcw, s).predict(Xi[etr]))
bag16 = np.clip(np.mean(np.array(preds),0),0,None)
print('8cfg x 2seed MAE on 431:', round(np.abs(bag16-ye).mean(),3), 'time', round(time.time()-t0,1))
for w in [0.6,0.7,0.8,1.0]:
    f = np.clip(w*bag16 + (1-w)*blend_feat[etr], 0, None)
    print(f'blend model*{w}+exp4w*{round(1-w,1)} MAE:', round(np.abs(f-ye).mean(),3))
