
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets(); train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')

tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
va = f[f.snapshot_day.isin(A.snapshot_days()['validation'])].copy()
print('train', len(tr), 'val', len(va))

Xf = tr[feat_cols].values.astype(np.float32); yf = tr.future_spend_4w.values
Xv = va[feat_cols].values.astype(np.float32)
sw = np.exp((tr.snapshot_day.values - 431) / 250.0)   # recency weighting, tau=250
CFGS = [(4,20),(4,40),(5,40),(6,40)]
BASE = dict(learning_rate=0.08, n_estimators=400, subsample=0.9, colsample_bytree=0.9,
            tree_method='hist', n_jobs=8)
t0 = time.time()
pb = np.zeros(len(va))
for d, mw in CFGS:
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                         max_depth=d, min_child_weight=mw, **BASE)
    m.fit(Xf, yf, sample_weight=sw); pb += m.predict(Xv) / len(CFGS)
print('base bag done %.0fs' % (time.time()-t0))

clf = xgb.XGBRegressor(objective='reg:logistic', max_depth=5, min_child_weight=40, **BASE)
clf.fit(Xf, (yf > 0).astype(float), sample_weight=sw)
pc = clf.predict(Xv)

pos = yf > 0
Xfp, yfp, swp = Xf[pos], yf[pos], sw[pos]
UP = np.array([0.1,0.2,0.3,0.4,0.5])
Q = np.zeros((len(UP), len(va)))
for i, u in enumerate(UP):
    acc = np.zeros(len(va))
    for d, mw in CFGS:
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=float(u),
                             max_depth=d, min_child_weight=mw, **BASE)
        m.fit(Xfp, yfp, sample_weight=swp); acc += m.predict(Xv) / len(CFGS)
    Q[i] = acc
print('quantile grid done %.0fs' % (time.time()-t0))

us = np.clip((pc - 0.5) / np.maximum(pc, 1e-6), 0.02, 0.5)
idx = np.clip((us - UP[0]) / (UP[1] - UP[0]), 0, len(UP) - 1)
lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int); w = idx - lo
mm = Q[lo, np.arange(len(pc))] * (1 - w) + Q[hi, np.arange(len(pc))] * w
mm = np.where(pc <= 0.5, 0.0, mm)

mix50 = 0.5 * pb + 0.5 * mm
e4w = va['exp4w_blend'].values.astype(float)
final = np.clip(0.9 * mix50 + 0.1 * e4w, 0, 5000)
print('final: mean %.1f med %.1f zeros %.3f' % (final.mean(), np.median(final), (final==0).mean()))

pred = va[['household_key','snapshot_day']].copy()
pred['prediction'] = final
path = save_table(pred, 'pred_e017.parquet')
print('saved', path, pred.shape)
