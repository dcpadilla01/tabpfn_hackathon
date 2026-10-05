
import pandas as pd, numpy as np

feats = load_saved('feats_v4.parquet')
print('feats_v4', feats.shape)
print(list(feats.columns))

names = ['pred_e005','pred_e007','pred_e011','pred_e013','pred_e014','pred_e015','pred_e008','pred_e012']
preds = {}
for nm in names:
    try:
        p = load_saved(nm + '.parquet')
        preds[nm] = p
        print(nm, p.shape, 'mean=%.1f med=%.1f' % (p['prediction'].mean(), p['prediction'].median()))
    except Exception as e:
        print(nm, 'ERR', type(e).__name__)

w = None
for nm, p in preds.items():
    q = p[['household_key','snapshot_day','prediction']].rename(columns={'prediction': nm})
    w = q if w is None else w.merge(q, on=['household_key','snapshot_day'])
print('wide preds', w.shape)
print(w.drop(columns=['household_key','snapshot_day']).corr().round(3).to_string())

tt = train_targets()
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','median']).round(1).to_string())


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A

tt = train_targets()
f = load_saved('feats_v4.parquet')
tr = f[f.snapshot_day.isin(A.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(tr), 'zero frac %.3f' % (tr.future_spend_4w==0).mean())

# trivial predictor MAEs on train
y = tr.future_spend_4w.values
for name, p in [('zero', np.zeros(len(y))), ('median', np.full(len(y), np.median(y))),
                ('lag1', tr.lag1_spend.values), ('exp4w_blend', tr.exp4w_blend.values),
                ('spend_28', tr.spend_28.values)]:
    print('%-12s MAE %.2f' % (name, np.abs(y-p).mean()))

# zero-inflation: by exp4w_blend bucket
tr['b'] = pd.qcut(tr.exp4w_blend.clip(lower=0), 10, duplicates='drop')
g = tr.groupby('b', observed=True).agg(n=('future_spend_4w','size'), zero=('future_spend_4w', lambda s:(s==0).mean()),
                                       med=('future_spend_4w','median'), mean=('future_spend_4w','mean'),
                                       med_lag=('lag1_spend','median'))
print(g.round(2).to_string())

# seasonal lag: spend in [snap-363, snap-336] (year-ago target window)
snap = A.snapshot(459)
tx = snap.transactions
txg = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
def yearago(day):
    lo, hi = day-363, day-336
    m = txg[(txg.day>=lo)&(txg.day<=hi)].groupby('household_key').sales_value.sum()
    return m
yl = {}
for d in [375,403,431]:
    yl[d] = yearago(d)
sub = tr[tr.snapshot_day.isin(yl.keys())].copy()
sub['ylag'] = [yl[d].get(h,0.0) for h,d in zip(sub.household_key, sub.snapshot_day)]
s = sub[sub.tenure>=364]
print('rows w/ full year-ago:', len(s), 'corr(ylag, y) = %.3f' % s[['ylag','future_spend_4w']].corr().iloc[0,1])
print('MAE lag1 %.2f | ylag %.2f | blend(0.5) %.2f | zero %.2f' % (
    np.abs(s.future_spend_4w-s.lag1_spend).mean(), np.abs(s.future_spend_4w-s.ylag).mean(),
    np.abs(s.future_spend_4w-0.5*(s.lag1_spend+s.ylag)).mean(), np.abs(s.future_spend_4w).mean()))
print('mean ylag %.1f vs mean y %.1f' % (s.ylag.mean(), s.future_spend_4w.mean()))


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A

# 1) aggregate spend by 4-week window over all households (view capped at 459)
snap = A.snapshot(459)
tx = snap.transactions
txg = tx.groupby('day').sales_value.sum()
tot = txg.reindex(range(1,460), fill_value=0)
w = pd.Series([tot.iloc[d-1:d+27].sum() for d in range(1, 433)], index=range(1,433))
print('4w-window total spend (all hh), selected windows:')
for d in [1,29,57,85,113,141,169,197,225,253,281,309,337,365,393,421]:
    print('  start %3d: %9.0f' % (d, w[d]))
print('mean of first 6 windows %.0f | windows 4-7 (95-179 starts: 85,113,141,169) %.0f | last 6 %.0f' % (
    w.iloc[:6].mean(), w.loc[[85,113,141,169]].mean(), w.iloc[-6:].mean()))
# seasonality ratio: window starting d vs d+364 impossible (data shorter); check within-year pattern via week
wk = txg.reindex(range(1,460), fill_value=0).groupby((np.arange(1,460)-1)//7).sum()
print('weekly total: first 13 wks %.0f, weeks 14-26 %.0f, last 13 wks %.0f' % (wk.iloc[:13].mean(), wk.iloc[13:26].mean(), wk.iloc[-13:].mean()))

# 2) E013 error breakdown on train (in-sample, indicative)
tt = train_targets()
f = load_saved('feats_v4.parquet')
tr = f[f.snapshot_day.isin(A.snapshot_days()['train'])].merge(tt, on=['household_key','snapshot_day'])
p13 = load_saved('pred_e013.parquet').rename(columns={'prediction':'p13'})
tr = tr.merge(p13, on=['household_key','snapshot_day'])
tr['err'] = (tr.future_spend_4w - tr.p13).abs()
tr['dec'] = pd.qcut(tr.p13, 10, duplicates='drop')
g = tr.groupby('dec', observed=True).agg(n=('err','size'), pred=('p13','mean'), y=('future_spend_4w','mean'),
                                         ymed=('future_spend_4w','median'), mae=('err','mean'))
print(g.round(1).to_string())
print('overall train MAE %.2f (in-sample)' % tr.err.mean())
# error by actual zero
for z in [0,1]:
    s = tr[tr.future_spend_4w==0] if z==0 else tr[tr.future_spend_4w>0]
    print('actual %s: n=%d MAE %.2f meanpred %.1f' % ('zero' if z==0 else '>0', len(s), s.err.mean(), s.p13.mean()))


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A

tt = train_targets()
f = load_saved('feats_v4.parquet')
p13 = load_saved('pred_e013.parquet')
print('p13 dtypes:', p13.dtypes.to_dict())
print('f dtypes:', f.dtypes.head(3).to_dict(), '| tt dtypes:', tt.dtypes.to_dict())
print('p13 head:'); print(p13.head(3))

p13 = p13.rename(columns={'prediction':'p13'})
m = f.merge(p13, on=['household_key','snapshot_day'], how='inner')
print('merged feats+pred:', len(m))
tr = m.merge(tt, on=['household_key','snapshot_day'])
print('merged with targets:', len(tr))
if len(tr):
    tr['err'] = (tr.future_spend_4w - tr.p13).abs()
    tr['dec'] = pd.qcut(tr.p13, 10, duplicates='drop')
    g = tr.groupby('dec', observed=True).agg(n=('err','size'), pred=('p13','mean'), y=('future_spend_4w','mean'),
                                             ymed=('future_spend_4w','median'), mae=('err','mean'))
    print(g.round(1).to_string())
    print('overall train MAE %.2f' % tr.err.mean())
    for z in [0,1]:
        s = tr[tr.future_spend_4w==0] if z==0 else tr[tr.future_spend_4w>0]
        print('actual %s: n=%d MAE %.2f meanpred %.1f' % ('zero' if z==0 else '>0', len(s), s.err.mean(), s.p13.mean()))


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A

snap = A.snapshot(459)
tx = snap.transactions
# stable cohort: first purchase <= day 94 (has rows at train snaps 95-179 AND late snaps)
first = tx.groupby('household_key').day.min()
cohort = first[first <= 94].index
print('cohort size:', len(cohort))
ctx = tx[tx.household_key.isin(cohort)]
cg = ctx.groupby(['household_key','day']).sales_value.sum().reset_index()

def win_spend(lo, hi):
    m = cg[(cg.day>=lo)&(cg.day<=hi)].groupby('household_key').sales_value.sum()
    return m.reindex(cohort).fillna(0.0)

# same cycle positions, early vs late: windows starting d and d+364 (364-day cycle)
pairs = [(85,449),(113,441),(141,433),(169,425),(197,397),(225,369)]
rows=[]
for e,l in pairs:
    a, b = win_spend(e,e+27), win_spend(l,l+27)
    rows.append((e, l, a.mean(), b.mean(), b.mean()/max(a.mean(),1e-9), (a==0).mean(), (b==0).mean()))
df = pd.DataFrame(rows, columns=['early_start','late_start','early_mean','late_mean','ratio','early_zero','late_zero'])
print(df.round(3).to_string())
print('avg ratio (late/early) = %.3f' % df.ratio.mean())

# also mid-period same positions: d+182
rows2=[]
for e in [85,113,141,169,197,225]:
    a, b = win_spend(e,e+27), win_spend(e+182,e+209)
    rows2.append((e, a.mean(), b.mean(), b.mean()/max(a.mean(),1e-9)))
print(pd.DataFrame(rows2, columns=['start','early','mid','ratio']).round(3).to_string())

# cohort spend by cycle position (day mod 364) across full history: mean 4w spend per window start every 28d
allstarts = list(range(85, 433, 28))
vals = [win_spend(d, d+27).mean() for d in allstarts]
s = pd.Series(vals, index=allstarts)
pos = (s.index - 1) % 364 + 1
sm = pd.DataFrame({'start': s.index, 'cycle_pos': pos, 'mean4w': s.values})
print(sm.round(1).to_string())


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.isotonic import IsotonicRegression

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
nobj = 0
for c in feat_cols:
    if not np.issubdtype(f[c].dtype, np.number):
        f[c] = pd.factorize(f[c])[0].astype('float32'); nobj += 1
print('non-numeric cols encoded:', nobj)

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
    sw = np.exp((tr.snapshot_day.values[fi] - 431) / 250.0) if recency else None
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


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')

tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
val = f[f.snapshot_day.isin(A.snapshot_days()['validation'])]
def X(df): return df[feat_cols].values.astype(np.float32)

BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
UP = [0.1, 0.2, 0.3, 0.4, 0.5]   # low quantiles of positive part

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
    Q = np.vstack([fit_q(Xf[pos], yf[pos], u).predict(Xe) for u in UP])  # 5 x n
    # corrected mixture median: u_pos = (0.5 - p0)/(1 - p0), clip [0.02, 0.5]
    us = np.clip((0.5 - pc) / (1 - pc), 0.02, 0.5)
    idx = np.clip((us - 0.1) / 0.1, 0, 4)  # knots at .1..0.5
    lo = np.floor(idx).astype(int); hi = np.ceil(idx).astype(int); w = idx - lo
    mixmed = Q[lo, np.arange(len(pc))] * (1 - w) + Q[hi, np.arange(len(pc))] * w
    mixmed = np.where(pc >= 0.5, 0.0, mixmed)
    res['mixmed'] = mixmed
    res['b_base_mix'] = 0.5 * res['base'] + 0.5 * mixmed
    res['b75_base'] = 0.75 * res['base'] + 0.25 * mixmed
    auc = roc_auc_score(ye > 0, pc)
    print('fold<=%d %.0fs AUC=%.3f meanpc=%.3f actualzero=%.3f' % (fit_max, time.time()-t0, auc, pc.mean(), (ye==0).mean()))
    return res, pc, ye

r1, pc1, ye1 = run_fold(403, [431])
r2, pc2, ye2 = run_fold(375, [403, 431])
e431 = tr.snapshot_day == 431; e2 = tr.snapshot_day.isin([403, 431])
names = ['base','mixmed','b_base_mix','b75_base']
print('\n%-12s %9s %9s %9s' % ('variant','f1(431)','f2','avg'))
for nm in names:
    a = np.abs(r1[nm] - tr.future_spend_4w.values[e431]).mean()
    b = np.abs(r2[nm] - tr.future_spend_4w.values[e2]).mean()
    print('%-12s %9.2f %9.2f %9.2f' % (nm, a, b, (a+b)/2))
# calibration of classifier on fold1 eval
dfc = pd.DataFrame({'pc': pc1, 'y0': (ye1==0).astype(int)})
dfc['b'] = pd.qcut(dfc.pc, 10, duplicates='drop')
print(dfc.groupby('b', observed=True).agg(meanpc=('pc','mean'), zerorate=('y0','mean'), n=('y0','size')).round(3).to_string())


# ---- cell ----

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


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')

# market aggregates from capped view (valid for all train snapshots <=431)
snap = A.snapshot(459)
tx = snap.transactions
day_tot = tx.groupby('day').sales_value.sum().reindex(range(1,460), fill_value=0.0)
day_hh  = tx.groupby('day').household_key.nunique().reindex(range(1,460), fill_value=0.0)
def mkt(d, lo_off, hi_off):
    lo, hi = d+lo_off, d+hi_off
    if lo < 1: return np.nan
    return day_tot.loc[lo:hi].sum()
def mkt_hh(d, lo_off, hi_off):
    lo, hi = d+lo_off, d+hi_off
    if lo < 1: return np.nan
    return day_hh.loc[lo:hi].sum()

mk = {}
for d in sorted(set(train_days)) + [459,487,515,543]:
    m28, m84, m168 = mkt(d,-27,0), mkt(d,-83,0), mkt(d,-167,0)
    h28 = mkt_hh(d,-27,0)
    mk[d] = dict(mkt28=m28, mkt84=m84, mkt168=m168,
                 mkt_mom=m28/(m84/3.0), mkt_hh28=h28, mkt_sph28=m28/max(h28,1),
                 mkt_yago=mkt(d,-363,-336))
mkdf = pd.DataFrame(mk).T.reset_index().rename(columns={'index':'snapshot_day'})
for c in mkdf.columns:
    if c!='snapshot_day': mkdf[c]=mkdf[c].astype('float32')
print(mkdf.round(2).to_string())

# household year-ago lag (NaN when unavailable)
hh_day = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
def ylag_map(d):
    lo,hi = d-363, d-336
    if lo < 1: return pd.Series(dtype=float)
    return hh_day[(hh_day.day>=lo)&(hh_day.day<=hi)].groupby('household_key').sales_value.sum()
yl = {d: ylag_map(d) for d in sorted(set(train_days))+[459,487,515,543]}

def add_cols(df):
    df = df.merge(mkdf, on='snapshot_day', how='left')
    yl_col = []
    for h,d in zip(df.household_key, df.snapshot_day):
        s = yl[d]
        yl_col.append(s.get(h, np.nan) if len(s) else np.nan)
    df['ylag'] = np.array(yl_col, dtype=float)
    df['ylag_ratio'] = df.ylag / (df.lag1_spend + 1.0)
    return df

f_mkt = add_cols(f.copy())
mc = ['mkt28','mkt84','mkt168','mkt_mom','mkt_hh28','mkt_sph28','mkt_yago','ylag','ylag_ratio']
for c in mc:
    if not pd.api.types.is_numeric_dtype(f_mkt[c]): f_mkt[c] = f_mkt[c].astype('float32')
print('\nmkt feature NaN counts on train snaps:', f_mkt[f_mkt.snapshot_day.isin(train_days)][mc].isna().sum().to_dict())

tr  = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
trm = f_mkt[f_mkt.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
def X(df, cols): return df[cols].values.astype(np.float32)
BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
def fit_q(Xtr, ytr, u):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr); return m

def run_fold(df, cols, fit_max, eval_days):
    fi = df.snapshot_day <= fit_max; ei = df.snapshot_day.isin(eval_days)
    Xf, yf = X(df[fi], cols), df.future_spend_4w.values[fi]
    Xe = X(df[ei], cols)
    return fit_q(Xf, yf, 0.5).predict(Xe)

fc = feat_cols; fcm = feat_cols + mc
res = {}
for tag, df, cols in [('v4', tr, fc), ('v4+mkt', trm, fcm), ('v4+mkt-noylag', trm, feat_cols+mc[:7])]:
    r1 = run_fold(df, cols, 403, [431]); r2 = run_fold(df, cols, 375, [403,431])
    e431 = tr.snapshot_day==431; e2 = tr.snapshot_day.isin([403,431])
    a = np.abs(r1-tr.future_spend_4w.values[e431]).mean(); b = np.abs(r2-tr.future_spend_4w.values[e2]).mean()
    res[tag] = (a,b,(a+b)/2)
    print('%-16s f1 %.2f  f2 %.2f  avg %.2f' % (tag,a,b,(a+b)/2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
from sklearn.isotonic import IsotonicRegression

f4 = load_saved('feats_v4.parquet').copy()
f3 = load_saved('feats_v3.parquet').copy()
print('f3 cols not in f4:', [c for c in f3.columns if c not in f4.columns])
tt = train_targets(); train_days = A.snapshot_days()['train']
def prep(df):
    df = df.copy()
    for c in df.columns:
        if c not in ('household_key','snapshot_day') and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.factorize(df[c])[0].astype('float32')
    return df
f4, f3 = prep(f4), prep(f3)
tr4 = f4[f4.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
tr3 = f3[f3.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
def fit_q(Xtr, ytr, u=0.5, sw=None):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr, sample_weight=sw); return m

def evalset(df, cols, fit_max, eval_days, recency_tau=None, iso_from=None):
    fi = df.snapshot_day <= fit_max; ei = df.snapshot_day.isin(eval_days)
    Xf, yf = df.loc[fi, cols].values.astype(np.float32), df.future_spend_4w.values[fi]
    Xe, ye = df.loc[ei, cols].values.astype(np.float32), df.future_spend_4w.values[ei]
    sd = df.snapshot_day.values[fi]
    sw = np.exp((sd - fit_max)/recency_tau) if recency_tau else None
    m = fit_q(Xf, yf, 0.5, sw); p = m.predict(Xe)
    if iso_from is not None:
        # calibrate using model trained on iso_from (earlier), predicting on eval days
        fj = df.snapshot_day <= iso_from
        mj = fit_q(df.loc[fj, cols].values.astype(np.float32), df.future_spend_4w.values[fj], 0.5)
        pj = mj.predict(Xe)
        iso = IsotonicRegression(out_of_bounds='clip').fit(pj, ye)
        p = iso.predict(p)
    return p, ye

cols4 = [c for c in tr4.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
cols3 = [c for c in tr3.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('n cols: f3 %d f4 %d' % (len(cols3), len(cols4)))
res = {}
for tag, df, cols in [('f3', tr3, cols3), ('f4', tr4, cols4)]:
    for tau in [None, 250, 150]:
        p, ye = evalset(df, cols, 403, [431], tau)
        p2, ye2 = evalset(df, cols, 375, [403,431], tau)
        res[(tag,tau)] = (np.abs(p-ye).mean(), np.abs(p2-ye2).mean())
for k,v in res.items():
    print('%-10s tau=%-4s f1 %.2f f2 %.2f avg %.2f' % (k[0], k[1], v[0], v[1], (v[0]+v[1])/2))
# isotonic variant on f4
p, ye = evalset(tr4, cols4, 403, [431], None, iso_from=375)
p2, ye2 = evalset(tr4, cols4, 375, [403,431], None, iso_from=347)
print('f4+iso      f1 %.2f f2 %.2f avg %.2f' % (np.abs(p-ye).mean(), np.abs(p2-ye2).mean(), (np.abs(p-ye).mean()+np.abs(p2-ye2).mean())/2))


# ---- cell ----

import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets(); train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')
tr = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
BASE = dict(learning_rate=0.08, n_estimators=400, subsample=0.9, colsample_bytree=0.9,
            tree_method='hist', n_jobs=8)
CFGS = [(4,20),(4,40),(5,40),(6,40)]
UP = np.array([0.1,0.2,0.3,0.4,0.5])
def mko(u, sw=None, obj='reg:quantileerror'):
    return xgb.XGBRegressor(objective=obj, quantile_alpha=u if obj=='reg:quantileerror' else 0.5, **BASE)

def run_fold(fit_max, eval_days, tau=None):
    fi = tr.snapshot_day <= fit_max; ei = tr.snapshot_day.isin(eval_days)
    Xf = tr.loc[fi, feat_cols].values.astype(np.float32); yf = tr.future_spend_4w.values[fi]
    Xe = tr.loc[ei, feat_cols].values.astype(np.float32); ye = tr.future_spend_4w.values[ei]
    e4w = tr.loc[ei,'exp4w_blend'].values
    sd = tr.snapshot_day.values[fi]
    sw = np.exp((sd-fit_max)/tau) if tau else None
    t0=time.time()
    pb = np.mean([mko(0.5, sw).fit(Xf,yf,sample_weight=sw).predict(Xe) for d,mw in CFGS
                  for m in [xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=d, min_child_weight=mw, **BASE)] ], axis=0)
    clf = xgb.XGBRegressor(objective='reg:logistic', max_depth=5, min_child_weight=40, **BASE)
    clf.fit(Xf,(yf>0).astype(float), sample_weight=sw); pc = clf.predict(Xe)
    pos = yf>0; Xfp = Xf[pos]; yfp = yf[pos]; swp = sw[pos] if tau else None
    Q = []
    for u in UP:
        Q.append(np.mean([xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, max_depth=d, min_child_weight=mw, **BASE
                                           ).fit(Xfp,yfp,sample_weight=swp).predict(Xe) for d,mw in CFGS], axis=0))
    Q = np.vstack(Q)
    us = np.clip((pc-0.5)/np.maximum(pc,1e-6), 0.02, 0.5)
    idx = np.clip((us-UP[0])/(UP[1]-UP[0]), 0, len(UP)-1)
    lo=np.floor(idx).astype(int); hi=np.ceil(idx).astype(int); w=idx-lo
    mm = Q[lo,np.arange(len(pc))]*(1-w) + Q[hi,np.arange(len(pc))]*w
    mm = np.where(pc<=0.5, 0.0, mm)
    out = {'base':pb, 'mix50':0.5*pb+0.5*mm, 'mix25':0.75*pb+0.25*mm}
    for nm, p in [('base',pb),('mix50',out['mix50'])]:
        for wt in [0.6,0.7,0.8,0.9,1.0]:
            out['%s_w%d'%(nm,int(wt*10))] = wt*p + (1-wt)*e4w
    print('fold<=%d tau=%s %.0fs' % (fit_max, tau, time.time()-t0))
    return out, ye

res = {}
for tau in [None, 250]:
    r1, y1 = run_fold(403, [431], tau)
    r2, y2 = run_fold(375, [403,431], tau)
    for nm in r1:
        a = np.abs(r1[nm]-y1).mean(); b = np.abs(r2[nm]-y2).mean()
        res[(nm,tau)] = (a+b)/2
w1 = pd.DataFrame({nm:v for (nm,t),v in res.items() if t is None}, index=['MAE']).T.sort_values('MAE')
w2 = pd.DataFrame({nm:v for (nm,t),v in res.items() if t==250}, index=['MAE']).T.sort_values('MAE')
print('\ntau=None:'); print(w1.round(2).to_string())
print('\ntau=250:'); print(w2.round(2).to_string())


# ---- cell ----

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
