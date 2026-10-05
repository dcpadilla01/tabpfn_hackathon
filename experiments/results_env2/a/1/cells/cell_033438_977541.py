import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
use_days = tr_days[:-2]
ref = 431
w = 0.5 ** ((ref - tr.snapshot_day)/140.0)
m = tr.snapshot_day.isin(use_days)
X = tr.loc[m, FEAT]; y = tr.loc[m,'future_spend_4w'].values; sw = w.loc[m].values
t0=time.time()
model = xgb.XGBRegressor(objective='reg:quantile', quantile_alpha=0.5, n_estimators=1200,
                         learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
model.fit(X, y, sample_weight=sw)
print('fit %.1fs' % (time.time()-t0))
tr['pred'] = model.predict(tr[FEAT])
oof = tr[tr.snapshot_day.isin(use_days)]
err = (oof.pred - oof.future_spend_4w).abs()
print('OOF MAE %.3f' % err.mean())
print(oof.groupby('snapshot_day').apply(lambda d: (d.pred-d.future_spend_4w).abs().mean(), include_groups=False).round(1).to_dict())
oof2 = oof.copy(); oof2['dec'] = pd.qcut(oof2.pred, 10, duplicates='drop')
cal = oof2.groupby('dec', observed=True).apply(lambda d: pd.Series({'pred':d.pred.mean(),'act':d.future_spend_4w.mean(),'n':len(d),'mae':(d.pred-d.future_spend_4w).abs().mean()}), include_groups=False)
print(cal.round(1))
agent_api.save_table(oof[['household_key','snapshot_day','pred','future_spend_4w']], 'oof_e5.parquet')