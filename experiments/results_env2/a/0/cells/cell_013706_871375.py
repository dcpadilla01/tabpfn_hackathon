
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
nobj = 0
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32'); nobj += 1
print('non-numeric cols encoded:', nobj)

tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
val = f[f.snapshot_day.isin(A.snapshot_days()['validation'])]

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
    clf = fit_clf(Xf, yf); pc = clf.predict(Xe); p0 = np.clip(pc, 0, 0.4999)
    pos = yf > 0
    if crude:
        q50 = fit_q(Xf[pos], yf[pos], 0.5).predict(Xe)
        return np.where(pc >= 0.5, 0.0, q50 * np.minimum((1 - p0) / 0.5, 1.0))
    knots = np.array([0.5] + UP)
    Q = np.vstack([fit_q(Xf[pos], yf[pos], u).predict(Xe) for u in knots])
    us = np.clip(0.5 / (1 - p0), 0.5, 0.9)
    out = np.zeros(len(p0))
    for i in np.where(pc < 0.5)[0]:
        out[i] = np.interp(us[i], knots, Q[:, i])
    return out

def run_fold(fit_max, eval_days, recency=False):
    fi = tr.snapshot_day <= fit_max
    ei = tr.snapshot_day.isin(eval_days)
    Xf, yf = X(tr[fi]), tr.future_spend_4w.values[fi]
    Xe, ye = X(tr[ei]), tr.future_spend_4w.values[ei]
    sw = np.exp((tr.snapshot_day.values[fi] - 431) / 255.0) if recency else None
    res = {}
    t0 = time.time()
    res['base'] = fit_q(Xf, yf, 0.5, sw).predict(Xe)
    res['m2a'] = two_part(Xf, yf, Xe)
    res['m2b'] = two_part(Xf, yf, Xe, crude=True)
    lg = xgb.XGBRegressor(objective='reg:squarederror', **BASE)
    lg.fit(Xf, np.log1p(yf), sample_weight=sw)
    res['m7log'] = np.expm1(lg.predict(Xe))
    res['b_base_m2a'] = 0.5 * res['base'] + 0.5 * res['m2a']
    res['b_base_m7'] = 0.5 * res['base'] + 0.5 * res['m7log']
    print('fold fit<=%d done %.0fs' % (fit_max, time.time() - t0))
    return res

r1 = run_fold(403, [431]); r2 = run_fold(375, [403, 431])
names = list(r1.keys())
e431 = tr.snapshot_day == 431
e2 = tr.snapshot_day.isin([403, 431])
print('\n%-12s %9s %9s %9s' % ('variant', 'f1(431)', 'f2', 'avg'))
for nm in names:
    a = np.abs(r1[nm] - tr.future_spend_4w.values[e431]).mean()
    b = np.abs(r2[nm] - tr.future_spend_4w.values[e2]).mean()
    print('%-12s %9.2f %9.2f %9.2f' % (nm, a, b, (a + b) / 2))
print('m2a exact-zero frac f1: %.3f' % (r1['m2a'] == 0).mean())
