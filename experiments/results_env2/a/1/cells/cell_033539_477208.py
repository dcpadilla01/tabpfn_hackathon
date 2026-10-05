import pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
tt = agent_api.train_targets()
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]
use_days = tr_days[:-2]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0)
m = tr.snapshot_day.isin(use_days)
model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=1200,
                         learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
model.fit(tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, sample_weight=w.loc[m].values)
tr['pred'] = model.predict(tr[FEAT])
# held-out late snapshots 403, 431: bias check
for d in [375, 403, 431]:
    s = tr[tr.snapshot_day==d]
    print(d, 'n=%d predmean %.1f actmean %.1f bias %.1f MAE %.1f' % (len(s), s.pred.mean(), s.future_spend_4w.mean(), s.pred.mean()-s.future_spend_4w.mean(), (s.pred-s.future_spend_4w).abs().mean()))
# multiplicative/additive correction search on 403+431 held out
h = tr[tr.snapshot_day.isin([403,431])]
base = (h.pred-h.future_spend_4w).abs().mean()
print('held403+431 base MAE %.2f' % base)
for c in [0.0,2.5,5,7.5,10,12.5,15,20]:
    print('  +%.1f -> %.2f' % (c, (h.pred+c-h.future_spend_4w).abs().mean()))
for f in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print('  x%.2f -> %.2f' % (f, (h.pred*f-h.future_spend_4w).abs().mean()))
# feature drift: mean spend_28 by snapshot (all 17 snaps incl val)
g = F.groupby('snapshot_day').spend_28.mean()
print('spend_28 mean by day:'); print(g.round(1).to_dict())
g2 = F.groupby('snapshot_day').spend_84.mean()
print('spend_84 mean by day:'); print(g2.round(1).to_dict())