
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')
tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
def X(df): return df[feat_cols].values.astype(np.float32)

BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
UP = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

def fit_q(Xtr, ytr, u):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr); return m

def run_fold(fit_max, eval_days):
    fi = tr.snapshot_day <= fit_max; ei = tr.snapshot_day.isin(eval_days)
    Xf, yf = X(tr[fi]), tr.future_spend_4w.values[fi]
    Xe, ye = X(tr[ei]), tr.future_spend_4w.values[ei]
    res = {}; t0=time.time()
    res['base'] = fit_q(Xf, yf, 0.5).predict(Xe)
    clf = xgb.XGBRegressor(objective='reg:logistic', **BASE)
    clf.fit(Xf, (yf > 0).astype(float))
    pc = clf.predict(Xe)
    pos = yf > 0
    Q = np.vstack([fit_q(Xf[pos], yf[pos], u).predict(Xe) for u in UP])
    us = np.clip((pc - 0.5) / np.maximum(pc, 1e-6), 0.02, 0.5)   # CORRECT: u of positive part
    idx = np.clip((us - UP[0]) / (UP[1] - UP[0]), 0, len(UP) - 1)
    lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int); w = idx - lo
    mm = Q[lo, np.arange(len(pc))] * (1 - w) + Q[hi, np.arange(len(pc))] * w
    mm = np.where(pc <= 0.5, 0.0, mm)                            # CORRECT mask
    res['mixmed'] = mm
    res['b_base_mix'] = 0.5 * res['base'] + 0.5 * mm
    res['b75_base'] = 0.75 * res['base'] + 0.25 * mm
    res['b25_mix'] = 0.75 * res['base'] + 0.25 * mm
    print('fold<=%d %.0fs  pc<=0.5 frac %.3f  mean mm %.1f  mean base %.1f' % (
        fit_max, time.time()-t0, (pc<=0.5).mean(), mm.mean(), res['base'].mean()))
    return res

r1 = run_fold(403, [431]); r2 = run_fold(375, [403, 431])
e431 = tr.snapshot_day == 431; e2 = tr.snapshot_day.isin([403, 431])
names = list(r1.keys())
print('\n%-12s %9s %9s %9s' % ('variant','f1(431)','f2','avg'))
for nm in names:
    a = np.abs(r1[nm] - tr.future_spend_4w.values[e431]).mean()
    b = np.abs(r2[nm] - tr.future_spend_4w.values[e2]).mean()
    print('%-12s %9.2f %9.2f %9.2f' % (nm, a, b, (a+b)/2))
