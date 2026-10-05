import pandas as pd, numpy as np, itertools

tt = train_targets().rename(columns={'future_spend_4w':'y'})

# ---------- A: OOF diagnostics for the e5-style model ----------
oof = load_saved('oof_e5.parquet')
o = oof.merge(tt, on=['household_key','snapshot_day'], suffixes=('_oof',''))
if 'future_spend_4w_oof' in o.columns:
    print('oof label consistency:', round(float(np.mean(o.future_spend_4w_oof==o.y)),4))
y = o['y'].values; p = o['pred'].values
print('OOF n=%d  MAE=%.3f  bias=%.3f  ymean=%.1f  pmean=%.1f' % (len(o), np.abs(p-y).mean(), (p-y).mean(), y.mean(), p.mean()))
per = o.groupby('snapshot_day').apply(lambda g: pd.Series({
    'mae': np.abs(g.pred-g['y']).mean(), 'bias': (g.pred-g['y']).mean(),
    'ymean': g['y'].mean(), 'pmean': g.pred.mean()}))
print(per.round(2).T)
best=(1e9,1.0,0.0)
for a in np.arange(0.90,1.121,0.01):
    for b in np.arange(-12,12.1,1.0):
        v = np.abs(a*p+b-y).mean()
        if v<best[0]: best=(v,a,b)
print('best affine on OOF: mae=%.3f  a=%.3f  b=%.2f  (base %.3f)' % (best[0],best[1],best[2],np.abs(p-y).mean()))
print('zero-target frac OOF: %.3f' % (y==0).mean())
print('target pctls:', np.percentile(y,[50,75,90,95,99]).round(1))

# ---------- B: true holdout 403/431 via e016 variants (trained on 95-375) ----------
ap = load_saved('e016_allpreds.parquet')
h = ap[ap.snapshot_day.isin([403,431])].merge(tt, on=['household_key','snapshot_day'])
print('\nholdout 403/431 rows:', len(h), 'ymean=%.1f' % h.y.mean())
vars_ = ['pq','pl','pc','pa']
for c in vars_:
    print(' %-3s MAE %.3f  bias %+.3f' % (c, np.abs(h[c]-h.y).mean(), (h[c]-h.y).mean()))
h['m4'] = h[vars_].mean(axis=1); h['med4'] = h[vars_].median(axis=1)
print('mean4 MAE %.3f   med4 MAE %.3f' % (np.abs(h.m4-h.y).mean(), np.abs(h.med4-h.y).mean()))
a,b = best[1], best[2]
for c in vars_+['m4']:
    print(' %-3s +OOF-cal MAE %.3f' % (c, np.abs(a*h[c]+b-h.y).mean()))
print(h[vars_].corr().round(4))
