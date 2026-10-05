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
etr = day==431
ye = y_full[etr]
print('day431 n:', ye.sum()>0, len(ye), 'mean', round(np.nanmean(ye),1), 'median', round(np.nanmedian(ye),1), 'zero frac', round((ye==0).mean(),3))
print('constant median MAE on 431:', round(np.abs(ye-np.nanmedian(y_full[day<=403])).mean(),3))
for f in ['exp4w_blend','spend_28','spend_84','lag1_spend']:
    v = feats[f].values[etr]
    print(f, 'MAE on 431:', round(np.abs(np.nan_to_num(v)-ye).mean(),3), 'corr:', round(np.corrcoef(np.nan_to_num(v), ye)[0,1],3))
# quick model again with more care
mtr = (day<=403) & ~np.isnan(y_full)
Xtr, ytr = Xi[mtr], y_full[mtr]
m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=4, min_child_weight=20,
                 subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
                 n_jobs=4, random_state=0, tree_method='hist')
t0=time.time(); m.fit(Xtr, ytr); pr = np.clip(m.predict(Xi[etr]),0,None)
print('q50 d4 MAE on 431:', round(np.abs(pr-ye).mean(),3), 'time', round(time.time()-t0,1))
print('pred stats:', np.round(np.percentile(pr,[10,50,90]),1), 'target stats:', np.round(np.percentile(ye,[10,50,90]),1))
# error by target bucket
b = pd.cut(ye, [-1,0,25,75,150,300,1e9])
df = pd.DataFrame({'ye':ye,'pr':pr,'b':b})
print(df.groupby('b', observed=True).agg(n=('ye','size'), med_t=('ye','median'), med_p=('pr','median'), mae=('ye', lambda s: np.abs(s-df.loc[s.index,'pr']).mean())))
