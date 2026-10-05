
import numpy as np, pandas as pd, agent_api
from sklearn.metrics import mean_absolute_error as mae

tt = agent_api.train_targets()
y = tt['future_spend_4w']
print('TRAIN TARGETS n=%d zero_share=%.3f' % (len(tt), (y==0).mean()))
print(y.describe(percentiles=[.5,.75,.9,.95,.99]).round(2).to_dict())
g = tt.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median'])
g['zero'] = tt.groupby('snapshot_day')['future_spend_4w'].apply(lambda s:(s==0).mean())
print(g.round(2))

oof = agent_api.load_saved('oof_e008.parquet')
print('OOF cols', oof.columns.tolist(), oof.shape)
pc = [c for c in oof.columns if c not in ('household_key','snapshot_day')]
m = tt.merge(oof, on=['household_key','snapshot_day'])
p = m[pc[0]].values.astype(float); t = m['future_spend_4w'].values.astype(float)
print('E008 OOF MAE %.3f  mean bias %.3f  med bias %.3f' % (mae(t,p), (p-t).mean(), np.median(p-t)))
bys = pd.DataFrame({'snap':m.snapshot_day,'t':t,'p':p,'ae':np.abs(p-t)}).groupby('snap').agg(tmean=('t','mean'),pmean=('p','mean'),mae=('ae','mean'))
print(bys.round(2))
print('--- post-hoc transforms on OOF ---')
for cap in [150,200,250,300,400,500,700,1000]:
    print(' cap %5.0f  MAE %.3f' % (cap, mae(t, np.minimum(p,cap))))
for fl in [0,5,10,20,30]:
    print(' floor %3.0f MAE %.3f' % (fl, mae(t, np.maximum(p,fl))))
for a in [0.85,0.9,0.95,1.0]:
    print(' pow %.2f MAE %.3f' % (a, mae(t, np.sign(p)*np.abs(p)**a)))
for s in [0.9,0.95,1.0,1.05]:
    print(' scale %.2f MAE %.3f' % (s, mae(t, p*s)))

p8 = agent_api.load_saved('pred_e008.parquet')
print('pred_e008 val stats', p8['prediction'].describe(percentiles=[.5,.9,.99]).round(2).to_dict())

# ---- log1p-target ensemble ----
sd = agent_api.snapshot_days()
tr_days, va_days = sd['train'], sd['validation']
feats = agent_api.load_saved('feats_v3.parquet')
fcols = [c for c in feats.columns if c not in ('household_key','snapshot_day')]
tr = feats[feats.snapshot_day.isin(tr_days)]
va = feats[feats.snapshot_day.isin(va_days)]
tt2 = tt.set_index(['household_key','snapshot_day'])
ytr = tt2.reindex(pd.MultiIndex.from_arrays([tr.household_key, tr.snapshot_day]))['future_spend_4w'].values
print('nan y:', np.isnan(ytr).sum(), 'Xtr', tr.shape, 'Xva', va.shape, 'n_feat', len(fcols))
Xtr = tr[fcols].to_numpy(dtype=np.float32); Xva = va[fcols].to_numpy(dtype=np.float32)
yltr = np.log1p(ytr)

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
vp=[]
for seed in (7,17):
    mdl = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=6, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, objective='reg:quantileerror', quantile_alpha=0.5,
        tree_method='hist', n_jobs=8, random_state=seed)
    mdl.fit(Xtr, yltr); vp.append(np.expm1(mdl.predict(Xva)))
print('xgb q50 x2 done')
mdl = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.02, max_depth=7, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist', n_jobs=8, random_state=11)
mdl.fit(Xtr, yltr); vp.append(np.expm1(mdl.predict(Xva)))
print('xgb sq done')
for seed in (3,13):
    h = HistGradientBoostingRegressor(loss='quantile', quantile=0.5, max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=seed)
    h.fit(Xtr, yltr); vp.append(np.expm1(h.predict(Xva)))
print('hgb x2 done')
pl = np.mean(vp, axis=0)
out = va[['household_key','snapshot_day']].copy(); out['prediction']=pl
path = agent_api.save_table(out, 'pred_log1p.parquet')
print('saved', path, out.shape, 'mean %.2f med %.2f zero %.3f' % (pl.mean(), np.median(pl), (pl<1).mean()))
