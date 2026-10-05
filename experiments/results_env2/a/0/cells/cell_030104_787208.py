
import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
key = ['household_key','snapshot_day']
# encode any categorical columns to integer codes
for c in feats.columns:
    if str(feats[c].dtype) == 'category' or feats[c].dtype == object:
        feats[c] = pd.factorize(feats[c].astype(str))[0]
val_days = [459,487,515,543]
trm = feats.snapshot_day.isin(sorted(tt.snapshot_day.unique()))
vm = feats.snapshot_day.isin(val_days)
tr = feats[trm].merge(tt, on=key)
print('train rows', len(tr), 'val rows', vm.sum())
y = tr.future_spend_4w.values
fcols = [c for c in feats.columns if c not in key]
Xtr = tr[fcols].values.astype(np.float32)
Xva = feats[vm][fcols].values.astype(np.float32)
CONFIGS = [(4,20),(5,40),(4,40),(6,60)]
med_y = float(np.median(y))
pv = np.zeros(len(Xva))
for seed in (0,1):
    for md, mcw in CONFIGS:
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md,
                             min_child_weight=mcw, learning_rate=0.08, n_estimators=400,
                             nthread=4, seed=seed, base_score=med_y)
        m.fit(Xtr, y)
        pv += m.predict(Xva)
pv /= (2*len(CONFIGS))
# honest global shift: median train-OOF residual
mg = tr[key+['future_spend_4w']].merge(oof, on=key)
shift = float(np.median(mg.future_spend_4w.values - mg.oof.values))
print('shift = %.3f' % shift)
pred = np.clip(pv + shift, 0, None)
out = feats.loc[vm, key].copy(); out['prediction'] = pred
print('out', out.shape, 'nan:', out.prediction.isna().sum(), 'mean %.2f median %.2f' % (out.prediction.mean(), out.prediction.median()))
print('val pred mean by day:'); print(out.groupby('snapshot_day').prediction.mean().round(2))
assert len(out)==9989 and out.prediction.notna().all()
A.save_table(out, 'pred_e020.parquet')
print('saved OK, elapsed %.0fs' % (time.time()-t0))
