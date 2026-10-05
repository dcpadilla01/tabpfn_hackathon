import pandas as pd, numpy as np, xgboost as xgb
print('xgb version:', xgb.__version__)

tt = train_targets().rename(columns={'future_spend_4w':'y'})
held = load_saved('e016_held.parquet').rename(columns={'future_spend_4w':'y'})
print('e016_held:', held.shape)
for c in ['sq0','sq1','sq2']:
    print(' %-4s MAE %.3f bias %+.2f mean %.1f zeros %.3f' % (c, np.abs(held[c]-held.y).mean(), (held[c]-held.y).mean(), held[c].mean(), (held[c]<=0).mean()))

ap = load_saved('e016_allpreds.parquet')
h = ap[ap.snapshot_day.isin([403,431])].merge(tt, on=['household_key','snapshot_day'])
for c in ['pq','pl','pa']:
    print(' %-3s MAE %.3f bias %+.2f mean %.1f zeros %.3f' % (c, np.abs(h[c]-h.y).mean(), (h[c]-h.y).mean(), h[c].mean(), (h[c]<=0).mean()))
print('\ncorr pq/pl/pa/sq*:')
print(h[['pq','pl','pa','sq0','sq1','sq2']].corr().round(3))
for d in [403,431]:
    g = h[h.snapshot_day==d]
    print('day %d: n=%d ymean=%.1f pq MAE %.2f bias %+.2f' % (d, len(g), g.y.mean(), np.abs(g.pq-g.y).mean(), (g.pq-g.y).mean()))

allF = load_saved('allF.parquet')
tr_days = list(range(95,376,28))
tr = allF[allF.snapshot_day.isin(tr_days)]
X = tr.drop(columns=['household_key','snapshot_day'])
print('\nallF train rows %d, feats %d, NaN %d, inf %d' % (len(tr), X.shape[1], X.isna().sum().sum(), np.isinf(X.values).sum()))
print('allF val rows:', len(allF[allF.snapshot_day.isin([403,431,459,487,515,543])]))

Xs = X.iloc[:2000,:20]
mi = pd.MultiIndex.from_frame(tr[['household_key','snapshot_day']])
ys = tt.set_index(['household_key','snapshot_day']).reindex(mi).values[:2000,0]
m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=50, max_depth=4, n_jobs=8)
m.fit(Xs, ys)
print('smoke ok, pred mean %.2f (y mean %.2f)' % (m.predict(Xs).mean(), ys.mean()))
