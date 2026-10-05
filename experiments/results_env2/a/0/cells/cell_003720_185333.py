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
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
Xe = Xi[etr]

configs = [(4,20),(4,40),(5,20),(5,40),(5,60),(6,40),(6,60),(4,60)]
def fit(alpha, depth, mcw, target):
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=alpha,
                     n_jobs=4, random_state=0, tree_method='hist')
    m.fit(Xtr, target); return m

t0=time.time()
raw=[]; lg=[]
for d,mcw in configs:
    raw.append(fit(0.5,d,mcw,ytr).predict(Xe))
    lg.append(np.expm1(fit(0.5,d,mcw,np.log1p(ytr)).predict(Xe)))
raw=np.clip(np.mean(np.array(raw),0),0,None)
lg=np.clip(np.mean(np.array(lg),0),0,None)
print('raw-scale 8-bag MAE on 431:', round(np.abs(raw-ye).mean(),3))
print('log-scale 8-bag MAE on 431:', round(np.abs(lg-ye).mean(),3), 'time', round(time.time()-t0,1))
for w in [0.25,0.5,0.75]:
    b=w*lg+(1-w)*raw
    print(f'blend w={w} log MAE:', round(np.abs(b-ye).mean(),3))
b=pd.cut(ye,[-1,0,25,75,150,300,1e9])
df=pd.DataFrame({'ye':ye,'raw':raw,'lg':lg,'b':b})
print(df.groupby('b',observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_raw=('raw','median'), med_log=('lg','median')))
