import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
feats = agent_api.load_saved('feats_v4.parquet'); tt = agent_api.train_targets()
cat_cols = [c for c in feats.columns if str(feats[c].dtype)=='category']
codes = feats.copy()
for c in cat_cols:
    codes[c] = codes[c].astype('object').fillna('__NA__').astype('category').cat.codes.astype('int32')+1
FEATS = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
X = codes[FEATS].astype('float32')
df = codes[['household_key','snapshot_day']].merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype('float32'); Xtr_all = X.loc[df.index].values
CONFIGS = [dict(n_estimators=400, learning_rate=0.08, max_depth=d, min_child_weight=mcw,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
           for d in (4,5,6) for mcw in (20,40,60)][:8]
TR = list(range(95,432,28))
tr_mask = df.snapshot_day.isin(TR).values
Xa, ya = Xtr_all[tr_mask], y[tr_mask]
sw = np.where(df.snapshot_day[tr_mask].values>=347, 1e4, 1.0)
val_mask = df.snapshot_day.isin([459,487,515,543]).values
Xv = Xtr_all[val_mask]
P = np.zeros((Xv.shape[0], len(CONFIGS)*2)); k=0
for cfg in CONFIGS:
    for sd in (0,1):
        p=dict(cfg); p['seed']=sd; p['quantile_alpha']=0.5; p['tree_method']='hist'; p['n_jobs']=4
        m=xgb.XGBRegressor(**p); m.fit(Xa,ya,sample_weight=sw)
        P[:,k]=m.predict(Xv); k+=1
pred = P.mean(axis=1)
out = df.loc[val_mask, ['household_key','snapshot_day']].copy()
out['prediction']=pred
print(out.shape, out.snapshot_day.value_counts().to_dict())
print(out.prediction.describe().round(3))
path = agent_api.save_table(out, 'pred_e016.parquet')
print('saved', path)