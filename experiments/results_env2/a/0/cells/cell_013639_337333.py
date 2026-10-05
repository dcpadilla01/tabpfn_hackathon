
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.isotonic import IsotonicRegression

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
nobj = 0
for c in feat_cols:
    if f[c].dtype == object:
        f[c] = pd.factorize(f[c])[0].astype('float32'); nobj += 1
print('object cols encoded:', nobj)

tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
val = f[f.snapshot_day.isin(A.snapshot_days()['validation'])]
print('train rows', len(tr), 'val rows', len(val))

def X(df): return df[feat_cols].values.astype(np.float32)

BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
UP = [0.6, 0.7, 0.8, 0.9]

def fit_q(Xtr, ytr, u, sw=None):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr, sample_weight=sw); return m

def fit_clf(Xtr, ytr):
    m = xgb.XGBRegressor(objective='reg:logistic', **BASE)
    m.fit(Xtr, (ytr > 0).astype(float)); return m

def two_part(Xf, yf, Xe, crude=False):
    clf = fit_clf(Xf, yf); p0 = np.clip(clf.predict(Xe), 0, 0.4999)
    pos = yf > 0
    if crude:
        q50 = fit_q(Xf[pos], yf[pos], 0.5).predict(Xe)
        return np.where(clf.predict(Xe) >= 0.5, 0.0, q50 * np.minimum((1 - p0) / 0.5, 1.0))
    knots = np.array([0.5] + UP)
    Q = np.vstack([fit_q(Xf[pos], yf[pos], u).predict(Xe) for u in knots])
    us = np.clip(0.5 / (1 - p0), 0.5, 0.9)
    out = np.zeros(len(p0))
    msk = clf.predict(Xe) < 0.5
    for i in np.where(msk)[0]:
        out[i] = np.interp(us[i], knots, Q[:, i])
    return out

def run_fold(fit_max, eval_days, recency=False):
    fi = tr.snapshot_day <= fit_max
    ei = tr.snapshot_day.isin(eval_days)
    Xf, yf, Xe, ye = X(tr[fi]), tr.future_spend_4w.values[fi], X(tr[ei]), tr.future_spend_4w.values[ei]
    sw = np.exp((tr.snapshot_day.values[fi] - 431) / 250.0) if recency else None
    res = {}
    t0 = time.time()
    m1 = fit_q(Xf, yf, 0.5, sw)
    res['base'] = m1.predict(Xe)
    res['m2a'] = two_part(Xf, yf, Xe)
    res['m2b'] = two_part(Xf, yf, Xe, crude=True)
    lg = xgb.XGBRegressor(objective='reg:squarederror', **BASE)
    lg.fit(Xf, np.log1p(yf), sample_weight=sw)
    res['m7log'] = np.expm1(lg.predict(Xe))
    res['b_base_m2a'] = 0.5 * res['base'] + 0.5 * res['m2a']
    res['b_base_m7'] = 0.5 * res['base'] + 0.5 * res['m7log']
    print('fold fit<=%d done %.0fs; frac p0>=0.5 on eval: %.3f' % (fit_max, time.time() - t0, (res['m2a'] == 0).mean()))
    return res, m1

out = {}
m1_f1, m1_f2 = None, None
r1, m1_f1 = run_fold(403, [431])
out['f1(431)'] = r1
r2, m1_f2 = run_fold(375, [403, 431])
out['f2(403,431)'] = r2
# isotonic: calibrate f1 preds@431 using f2 model preds@431 (out-of-sample for both)
e431 = tr.snapshot_day == 431
p_f2_on_431 = m1_f2.predict(X(tr[e431])); y431 = tr.future_spend_4w.values[e431]
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_f2_on_431, y431)
r1['m3iso'] = iso.predict(r1['base'])
out['f1(431)'] = r1

names = list(out['f1(431)'].keys())
print('\n%-12s %9s %9s %9s' % ('variant', 'f1(431)', 'f2', 'avg'))
for nm in names:
    a = np.abs(out['f1(431)'][nm] - tr.future_spend_4w.values[e431]).mean()
    e2 = tr.snapshot_day.isin([403, 431])
    b = np.abs(out['f2(403,431)'][nm] - tr.future_spend_4w.values[e2]).mean()
    print('%-12s %9.2f %9.2f %9.2f' % (nm, a, b, (a + b) / 2))
print('base zero-frac check: pred==0 frac f1: %.3f' % (out['f1(431)']['base'] == 0).mean())
# diagnostic: fold1 model mean pred at 431 vs validation snapshots
vd = val.copy(); vd['p'] = m1_f1.predict(X(val))
print('\nfold1-model mean pred by snapshot: 431: %.1f | val:' % vd[vd.snapshot_day==431].p.mean() if False else '')
p431 = m1_f1.predict(X(tr[e431]))
print('fold1-model mean pred @431: %.1f' % p431.mean())
print(vd.groupby('snapshot_day').p.agg(['mean','median']).round(1).to_string())
