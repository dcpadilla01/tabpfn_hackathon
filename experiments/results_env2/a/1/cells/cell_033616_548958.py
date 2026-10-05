import pandas as pd, numpy as np
oof = agent_api.load_saved('oof_e5.parquet')
oof['resid'] = oof.future_spend_4w - oof.pred
print('OOF median(act-pred) = %.2f  mean = %.2f' % (oof.resid.median(), oof.resid.mean()))
oof['dec'] = pd.qcut(oof.pred, 10, duplicates='drop')
med = oof.groupby('dec', observed=True).agg(pred=('pred','mean'), act_med=('future_spend_4w','median'), act_mean=('future_spend_4w','mean'), resid_med=('resid','median'), n=('resid','size'))
print(med.round(1))
# global shift transfer test on held-out 403/431
F = agent_api.load_saved('allF.parquet'); tt = agent_api.train_targets()
tr = tt.merge(F.drop(columns=['future_spend_4w']), on=['household_key','snapshot_day'], how='left')
# recompute preds for 403/431 using saved e5 preds? e5 preds are val only. Use oof mapping applied to held-out preds:
# first retrain quickly to get preds at 403/431 (same config as before)
import xgboost as xgb, time
FEAT = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
use_days = [95,123,151,179,207,235,263,291,319,347,375]
w = 0.5 ** ((431 - tr.snapshot_day)/140.0); m = tr.snapshot_day.isin(use_days)
model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=1200, learning_rate=0.03,
    max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, tree_method='hist', n_jobs=8)
t0=time.time(); model.fit(tr.loc[m,FEAT], tr.loc[m,'future_spend_4w'].values, sample_weight=w.loc[m].values); print('fit %.0fs'%(time.time()-t0))
tr['pred'] = model.predict(tr[FEAT])
h = tr[tr.snapshot_day.isin([403,431])].copy()
print('held base MAE %.2f' % (h.pred-h.future_spend_4w).abs().mean())
print('held median(act-pred) = %.2f' % (h.future_spend_4w-h.pred).median())
# apply OOF per-decile median mapping (bin edges from OOF pred)
bins = oof.groupby('dec', observed=True)['pred'].mean().values
edges = np.unique(pd.qcut(oof.pred, 10, duplicates='drop', retbins=True)[1])
h['bin'] = pd.cut(h.pred, edges, include_lowest=True)
mapping = oof.groupby('dec', observed=True).apply(lambda d: d.future_spend_4w.median() - d.pred.mean(), include_groups=False)
h['map_pred'] = h.pred + h['bin'].map(mapping).fillna(0)
print('held MAE after per-decile median mapping: %.2f' % (h.map_pred-h.future_spend_4w).abs().mean())
for c in [0,5,10,15,20,25]:
    print('  held +%.0f -> %.2f' % (c, (h.pred+c-h.future_spend_4w).abs().mean()))
# also OOF MAE after mapping
oof['map_pred'] = oof.pred + oof['dec'].map(mapping).fillna(0)
print('OOF MAE after mapping: %.2f (base %.2f)' % ((oof.map_pred-oof.future_spend_4w).abs().mean(), (oof.pred-oof.future_spend_4w).abs().mean()))