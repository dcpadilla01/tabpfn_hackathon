
import pandas as pd, numpy as np, time
import agent_api as api
import xgboost as xgb

feats = api.load_saved('feats_v3.parquet')
print('feats', feats.shape)
tt = api.train_targets()
df = feats.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feat_cols = [c for c in feats.columns if c not in drop]
print('rows', len(df), 'n_feat', len(feat_cols), 'days', sorted(df.snapshot_day.unique()))
obj_cols = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
print('non-numeric cols:', obj_cols[:10])
for c in obj_cols:
    if str(df[c].dtype) == 'category': df[c] = df[c].cat.codes
    else: df[c] = pd.factorize(df[c])[0]

holdout = [347, 375, 403, 431]
inner = [d for d in sorted(df.snapshot_day.unique()) if d not in holdout]
m_tr = df.snapshot_day.isin(inner).values
m_ho = df.snapshot_day.isin(holdout).values
Xtr = df.loc[m_tr, feat_cols].astype(float).values
Xho = df.loc[m_ho, feat_cols].astype(float).values
ytr = df.loc[m_tr,'future_spend_4w'].values
yho = df.loc[m_ho,'future_spend_4w'].values
print('train', Xtr.shape, 'holdout', Xho.shape, 'holdout zero frac', round((yho==0).mean(),3))

base = dict(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.8, random_state=7, n_jobs=8)

def fit_pred(params, Xtr, ytr, Xho, post=None):
    m = xgb.XGBRegressor(**params); m.fit(Xtr, ytr)
    p = m.predict(Xho)
    if post is not None: p = post(p)
    return np.clip(p, 0, None)

t0=time.time()
oof_sq = fit_pred({**base,'objective':'reg:squarederror'}, Xtr, ytr, Xho)
print('sq  MAE', round(np.abs(oof_sq-yho).mean(),3), round(time.time()-t0,1),'s')

t0=time.time()
oof_med = fit_pred({**base,'n_estimators':1500,'learning_rate':0.03,'max_depth':6,
                    'objective':'reg:quantileerror','quantile_alpha':0.5}, Xtr, ytr, Xho)
print('med MAE', round(np.abs(oof_med-yho).mean(),3), round(time.time()-t0,1),'s')

t0=time.time()
oof_log = fit_pred({**base,'objective':'reg:squarederror'}, Xtr, np.log1p(ytr), Xho, post=np.expm1)
print('log MAE', round(np.abs(oof_log-yho).mean(),3), round(time.time()-t0,1),'s')

print('e006-approx MAE', round(np.abs(0.5*oof_sq+0.5*oof_med-yho).mean(),3))
rows=[]
for a in np.arange(0,1.0001,0.05):
    for b in np.arange(0,1.0001-a+1e-9,0.05):
        c = 1-a-b
        if c < -1e-9: continue
        mae = np.abs(a*oof_sq+b*oof_med+c*oof_log-yho).mean()
        rows.append((round(a,2),round(b,2),round(c,2),round(mae,3)))
rows.sort(key=lambda r:r[3])
print('top weight combos (sq,med,log,mae):')
for r in rows[:12]: print(r)

oof_df = pd.DataFrame({'household_key': df.loc[m_ho,'household_key'].values,
                       'snapshot_day': df.loc[m_ho,'snapshot_day'].values,
                       'oof_sq':oof_sq,'oof_med':oof_med,'oof_log':oof_log})
print('saved:', api.save_table(oof_df,'oof_e008'))

for n in ['pred_e004','pred_e005','pred_e006']:
    p = api.load_saved(n)
    print(n, p.shape, p.columns.tolist())
