import pandas as pd, numpy as np

names = ['e001_preds','e002_preds','e003_preds','e004_preds','e005_preds','e007_preds','e008_preds',
         'e009_preds','e010_preds','e011_preds','e012_preds','e014_preds','e016_preds','e017_preds',
         'e018_preds','e019_blend_preds']
tabs = {}
for n in names:
    try:
        df = load_saved(n + '.parquet')
        tabs[n] = df
    except Exception as e:
        print(n, 'ERR', type(e).__name__, str(e)[:80])

ref = tabs['e019_blend_preds']
print('ref rows', ref.shape, 'days', sorted(ref.snapshot_day.unique()))
print('dup keys:', ref.duplicated(['household_key','snapshot_day']).sum())

P = ref[['household_key','snapshot_day']].copy()
for n, df in tabs.items():
    t = df.rename(columns={'prediction': n})[['household_key','snapshot_day', n]]
    P = P.merge(t, on=['household_key','snapshot_day'], how='left')
cols = [n for n in tabs if n in P.columns]
print('merged', P.shape)
print('NaN counts per model:'); print(P[cols].isna().sum())

print('\n--- prediction distribution per model ---')
print(P[cols].describe().T[['mean','std','min','50%','max']].round(2))

print('\n--- correlation with e019 blend ---')
C = P[cols].corr()
print(C['e019_blend_preds'].round(4).sort_values())

print('\n--- mean abs deviation from e019 blend ---')
print(P[cols].sub(P['e019_blend_preds'], axis=0).abs().mean().round(2).sort_values())

print('\n--- train targets ---')
tt = train_targets()
print(tt.shape)
print(tt.groupby('snapshot_day').future_spend_4w.agg(['count','mean','median']).round(1))
print(tt.future_spend_4w.describe().round(2))

print('\n--- feature tables available ---')
for f in ['allF','e004_features','e002_features','e005_newfeats','camp_feats','lagfeats','lagfeats2','f_weekly']:
    try:
        d = load_saved(f + '.parquet')
        print(f, d.shape, 'cols:', list(d.columns)[:6], '... days:', sorted(d.snapshot_day.unique())[:20])
    except Exception as e:
        print(f, 'ERR', type(e).__name__, str(e)[:60])


# ---- cell ----
import pandas as pd, numpy as np

for f in ['oof_e5','repro_e5','e016_allpreds','e016_sqpreds','e016_held','e004_new','e004_preds','e005_preds','e011_preds','e018_preds']:
    try:
        d = load_saved(f + '.parquet')
        print('===', f, d.shape)
        print(' cols:', list(d.columns))
        if 'snapshot_day' in d.columns:
            print(' days:', sorted(d.snapshot_day.unique()))
    except Exception as e:
        print(f, 'ERR', type(e).__name__, str(e)[:80])

# train targets coverage
tt = train_targets()
print('\ntarget days:', sorted(tt.snapshot_day.unique()))
print('rows per day:'); print(tt.groupby('snapshot_day').size())


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np, xgboost
print('xgb version:', xgboost.__version__)

tt = train_targets().rename(columns={'future_spend_4w':'y'})
held = load_saved('e016_held.parquet')
print('\ne016_held:', held.shape)
for c in ['sq0','sq1','sq2']:
    print(' %-4s MAE %.3f bias %+.2f mean %.1f zeros %.3f' % (c, np.abs(held[c]-held.y).mean(), (held[c]-held.y).mean(), held[c].mean(), (held[c]<=0).mean()))

ap = load_saved('e016_allpreds.parquet')
h = ap[ap.snapshot_day.isin([403,431])].merge(tt, on=['household_key','snapshot_day'])
for c in ['pq','pl','pa']:
    print(' %-3s MAE %.3f bias %+.2f mean %.1f zeros %.3f' % (c, np.abs(h[c]-h.y).mean(), (h[c]-h.y).mean(), h[c].mean(), (h[c]<=0).mean()))
print('\ncorr sq* vs pq/pa:')
print(h[['pq','pl','pa','sq0','sq1','sq2']].corr().round(3))

# per-snapshot holdout MAE of pq (stability check)
for d in [403,431]:
    g = h[h.snapshot_day==d]
    print('day %d: n=%d ymean=%.1f pq MAE %.2f bias %+.2f' % (d, len(g), g.y.mean(), np.abs(g.pq-g.y).mean(), (g.pq-g.y).mean()))

# feature tables sanity
allF = load_saved('allF.parquet')
tr_days = list(range(95,376,28))
tr = allF[allF.snapshot_day.isin(tr_days)]
X = tr.drop(columns=['household_key','snapshot_day'])
print('\nallF train rows %d, feats %d, NaN %d, inf %d' % (len(tr), X.shape[1], X.isna().sum().sum(), np.isinf(X.values).sum()))
print('allF val rows:', len(allF[allF.snapshot_day.isin([403,431,459,487,515,543])]))
e2 = load_saved('e002_features.parquet')
print('e002_features cols:', e2.shape[1], 'NaN:', e2.drop(columns=['household_key','snapshot_day','index'], errors='ignore').isna().sum().sum())

# quick smoke test of quantile objective
import xgboost as xgb
Xs = X.iloc[:2000,:20]; ys = tt.set_index(['household_key','snapshot_day']).reindex(pd.MultiIndex.from_frame(tr[['household_key','snapshot_day']])).values[:2000,0]
m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=50, max_depth=4, n_jobs=8)
m.fit(Xs, ys)
print('smoke test ok, pred mean %.2f (y mean %.2f)' % (m.predict(Xs).mean(), ys.mean()))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb
print('xgb version:', xgboost.__version__)

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


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np
from scipy.optimize import minimize

tt = train_targets().rename(columns={'future_spend_4w':'y'})
ap = load_saved('e016_allpreds.parquet')
held = load_saved('e016_held.parquet').rename(columns={'future_spend_4w':'y'})
h = held[['household_key','snapshot_day','sq0','sq1','sq2','y']].merge(
    ap[['household_key','snapshot_day','pq','pl','pa']], on=['household_key','snapshot_day'], how='inner')
print('holdout merged:', h.shape)
mem = ['pq','pl','pa','sq0','sq1','sq2']
P = h[mem].values; y = h['y'].values
for i,c in enumerate(mem):
    print(' %-4s MAE %.3f' % (c, np.abs(P[:,i]-y).mean()))

def mae_w(w, P, y):
    return np.abs(P @ w - y).mean()

# pairwise blends
print('\npair blends (0.5/0.5):')
for i in range(len(mem)):
    for j in range(i+1, len(mem)):
        m = np.abs(0.5*P[:,i]+0.5*P[:,j]-y).mean()
        if m < 62.0: print('  %s+%s: %.3f' % (mem[i],mem[j],m))

# weight fit with day cross-validation
d403 = (h.snapshot_day==403).values; d431 = ~d403
def fit_w(mask):
    Pm, ym = P[mask], y[mask]
    f = lambda z: mae_w(np.exp(z)/np.exp(z).sum(), Pm, ym)
    r = minimize(f, np.zeros(len(mem)), method='Nelder-Mead',
                 options={'maxiter':4000,'xatol':1e-4,'fatol':1e-5})
    return np.exp(r.x)/np.exp(r.x).sum()
w403 = fit_w(d403); w431 = fit_w(d431); wall = fit_w(np.ones(len(h),bool))
print('\nweights fit@403 :', dict(zip(mem, w403.round(3))), ' -> MAE@431 %.3f' % mae_w(w403, P[d431], y[d431]))
print('weights fit@431 :', dict(zip(mem, w431.round(3))), ' -> MAE@403 %.3f' % mae_w(w431, P[d403], y[d403]))
print('weights fit@all :', dict(zip(mem, wall.round(3))), ' -> MAE@all %.3f (in-sample)' % mae_w(wall, P, y))
print('equal weights    : MAE@all %.3f' % mae_w(np.ones(6)/6, P, y))
print('pq alone         : MAE@all %.3f' % np.abs(h.pq-y).mean())

# pq vs pq+sq0 weight curve
print('\npq/sq0 mix curve:')
for w in [0,.1,.2,.3,.4,.5,.6,.7,.8,1.0]:
    p = w*h.pq + (1-w)*h.sq0
    print('  w_pq=%.1f  MAE %.3f' % (w, np.abs(p-y).mean()))

# affine cal transfer for pq and best blend
for name, p in [('pq', h.pq), ('pq+sq0 .5', .5*h.pq+.5*h.sq0)]:
    a403 = np.polyfit(p[d403], y[d403], 1)
    p431 = a403[0]*p[d431]+a403[1]
    print('%s cal@403->431: %.3f (uncal %.3f)' % (name, np.abs(p431-y[d431]).mean(), np.abs(p[d431]-y[d431]).mean()))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

allF = load_saved('allF.parquet')
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day')]
X_all = allF[FEATS].astype(np.float64).replace([np.inf,-np.inf], np.nan)
print('feats:', len(FEATS), 'NaN frac: %.4f' % X_all.isna().values.mean())

tt = train_targets().rename(columns={'future_spend_4w':'y'})
y_map = tt.set_index(['household_key','snapshot_day'])['y']

def make(ds):
    m = allF.snapshot_day.isin(ds).values
    X = X_all[m]; idx = allF.loc[m, ['household_key','snapshot_day']]
    y = y_map.reindex(pd.MultiIndex.from_frame(idx)).values
    w = 0.5 ** ((max(ds) - idx.snapshot_day.values) / 140.0)
    return X, y, w, idx

def fit_pred(ds_tr, ds_pred, seeds, alpha=0.5, rounds=1200):
    Xtr, ytr, wtr, _ = make(ds_tr)
    Xp, _, _, idxp = make(ds_pred)
    P = np.zeros((len(Xp), len(seeds)))
    for k, s in enumerate(seeds):
        t0 = time.time()
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                             n_estimators=rounds, learning_rate=0.03, max_depth=6,
                             min_child_weight=25, subsample=0.7, tree_method='hist',
                             n_jobs=16, random_state=s)
        m.fit(Xtr, ytr, sample_weight=wtr)
        P[:, k] = np.clip(m.predict(Xp), 0, None)
        print('  seed %d done %.0fs' % (s, time.time()-t0), flush=True)
    return P.mean(axis=1), idxp

D11 = list(range(95,376,28))
hold_days = [403,431]
h = allF[allF.snapshot_day.isin(hold_days)][['household_key','snapshot_day']].copy()
h['y'] = y_map.reindex(pd.MultiIndex.from_frame(h)).values
pq = load_saved('e016_allpreds.parquet')
h = h.merge(pq[['household_key','snapshot_day','pq']], on=['household_key','snapshot_day'])
yv = h['y'].values
print('holdout rows', len(h), 'pq MAE %.3f' % np.abs(h.pq.values-yv).mean())

t0=time.time()
p11, idx11 = fit_pred(D11, hold_days, [7,8])
h2 = h.merge(idx11.assign(p11=p11), on=['household_key','snapshot_day'], how='left')
p11a = h2.p11.values
print('C11 (11-snap, 2 seeds) MAE: all %.3f | 403 %.3f | 431 %.3f | total %.0fs' % (
    np.abs(p11a-yv).mean(), np.abs(p11a[h2.snapshot_day==403]-yv[h2.snapshot_day==403]).mean(),
    np.abs(p11a[h2.snapshot_day==431]-yv[h2.snapshot_day==431]).mean(), time.time()-t0))
print('corr(p11,pq) %.4f' % np.corrcoef(p11a, h2.pq.values)[0,1])
for w in [0,.2,.4,.5,.6,.8,1.0]:
    p = w*p11a + (1-w)*h2.pq.values
    print('  w_C11=%.1f MAE %.3f' % (w, np.abs(p-yv).mean()))


# ---- cell ----
import pandas as pd, numpy as np

allF = load_saved('allF.parquet')
print('ALL COLUMNS (%d):' % (len(allF.columns)))
print(list(allF.columns))

tt = train_targets().rename(columns={'future_spend_4w':'y'})
h = allF[allF.snapshot_day.isin([403,431])].merge(tt, on=['household_key','snapshot_day'])
print('\nholdout rows:', len(h), 'ymean %.1f' % h.y.mean())
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day')]
corrs = {}
for c in FEATS:
    v = h[c].astype(float)
    if v.notna().sum() > 100:
        corrs[c] = np.corrcoef(v.fillna(v.mean()), h.y)[0,1]
top = pd.Series(corrs).abs().sort_values(ascending=False).head(15)
print('\ntop |corr| with y at 403/431:')
print(top.round(4))

# exact-match check: any column == y?
for c in FEATS:
    v = h[c]
    if v.notna().equals(h.y.notna()) and np.allclose(v.fillna(-1), h.y.fillna(-1)):
        print('EXACT MATCH WITH Y:', c)

# NaN pattern by snapshot day for the top-correlated column
c0 = top.index[0]
print('\nNaN rate of %s by snapshot_day:' % c0)
print(allF.groupby('snapshot_day')[c0].apply(lambda s: s.isna().mean()).round(3).to_string())


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

allF = load_saved('allF.parquet')
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X_all = allF[FEATS].astype(np.float64).replace([np.inf,-np.inf], np.nan)
tt = train_targets().rename(columns={'future_spend_4w':'y'})
y_map = tt.set_index(['household_key','snapshot_day'])['y']

def make(ds):
    m = allF.snapshot_day.isin(ds).values
    X = X_all[m]; idx = allF.loc[m, ['household_key','snapshot_day']]
    y = y_map.reindex(pd.MultiIndex.from_frame(idx)).values
    return X, y, idx

def fit_pred(ds_tr, ds_pred, seeds, decay=140.0, rounds=1200):
    Xtr, ytr, idxtr = make(ds_tr)
    wtr = 0.5 ** ((max(ds_tr) - idxtr.snapshot_day.values) / decay) if decay else None
    Xp, _, idxp = make(ds_pred)
    P = np.zeros((len(Xp), len(seeds)))
    for k, s in enumerate(seeds):
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                             n_estimators=rounds, learning_rate=0.03, max_depth=6,
                             min_child_weight=25, subsample=0.7, tree_method='hist',
                             n_jobs=16, random_state=s)
        m.fit(Xtr, ytr, sample_weight=wtr)
        P[:, k] = np.clip(m.predict(Xp), 0, None)
    return P.mean(axis=1), idxp

D11 = list(range(95,376,28))
hold = [403,431]
h = allF[allF.snapshot_day.isin(hold)][['household_key','snapshot_day']].copy()
h['y'] = y_map.reindex(pd.MultiIndex.from_frame(h)).values
ap = load_saved('e016_allpreds.parquet'); sq = load_saved('e016_sqpreds.parquet')
h = h.merge(ap[['household_key','snapshot_day','pq','pl','pa']], on=['household_key','snapshot_day'])
h = h.merge(sq[['household_key','snapshot_day','sq0']], on=['household_key','snapshot_day'])
yv = h.y.values
print('holdout rows %d | pq %.3f | sq0 %.3f | pq+sq0 .8 %.3f' % (
    len(h), np.abs(h.pq-yv).mean(), np.abs(h.sq0-yv).mean(),
    np.abs(0.8*h.pq+0.2*h.sq0-yv).mean()))

t0=time.time()
p140, i140 = fit_pred(D11, hold, [7,8], decay=140.0)
h = h.merge(i140.assign(c140=p140), on=['household_key','snapshot_day'], how='left')
print('C11 d140 (2 seeds): MAE %.3f  [403 %.3f | 431 %.3f]  corr(pq) %.4f' % (
    np.abs(h.c140-yv).mean(), np.abs(h.c140[h.snapshot_day==403]-yv[h.snapshot_day==403]).mean(),
    np.abs(h.c140[h.snapshot_day==431]-yv[h.snapshot_day==431]).mean(),
    np.corrcoef(h.c140, h.pq)[0,1]))

pnd, ind = fit_pred(D11, hold, [7,8], decay=0.0)
h = h.merge(ind.assign(cnd=pnd), on=['household_key','snapshot_day'], how='left')
print('C11 nodec (2 seeds): MAE %.3f  [403 %.3f | 431 %.3f]' % (
    np.abs(h.cnd-yv).mean(), np.abs(h.cnd[h.snapshot_day==403]-yv[h.snapshot_day==403]).mean(),
    np.abs(h.cnd[h.snapshot_day==431]-yv[h.snapshot_day==431]).mean()))

print('\nblends with pq (2-seed C11 d140):')
for w in [0,.2,.3,.4,.5,.6,.7,.8,1.0]:
    p = w*h.c140 + (1-w)*h.pq
    print('  w_C11=%.1f  MAE %.3f' % (w, np.abs(p-yv).mean()))
print('3-way pq/c140/sq0 (w .7/.2/.1): %.3f' % np.abs(0.7*h.pq+0.2*h.c140+0.1*h.sq0-yv).mean())
print('total time %.0fs' % (time.time()-t0))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

allF = load_saved('allF.parquet')
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X_all = allF[FEATS].astype(np.float64).replace([np.inf,-np.inf], np.nan)
tt = train_targets().rename(columns={'future_spend_4w':'y'})
y_map = tt.set_index(['household_key','snapshot_day'])['y']

def make(ds):
    m = allF.snapshot_day.isin(ds).values
    idx = allF.loc[m, ['household_key','snapshot_day']]
    return X_all[m], y_map.reindex(pd.MultiIndex.from_frame(idx)).values, idx

def run(ds_tr, ds_pred, seeds, decay=140.0, rounds=1200, alpha=0.5):
    Xtr, ytr, itr = make(ds_tr)
    w = 0.5 ** ((max(ds_tr)-itr.snapshot_day.values)/decay) if decay else None
    Xp, _, ip = make(ds_pred)
    P = np.zeros((len(Xp), len(seeds)))
    for k,s in enumerate(seeds):
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
                             n_estimators=rounds, learning_rate=0.03, max_depth=6,
                             min_child_weight=25, subsample=0.7, tree_method='hist',
                             n_jobs=16, random_state=s)
        m.fit(Xtr, ytr, sample_weight=w)
        P[:,k] = np.clip(m.predict(Xp), 0, None)
    return P.mean(axis=1), ip

D10 = list(range(95,348,28)); D9 = D10[:-1]
PV = [375,403,431]
base = allF[allF.snapshot_day.isin(PV)][['household_key','snapshot_day']].copy()
base['y'] = y_map.reindex(pd.MultiIndex.from_frame(base)).values
yv = base.y.values
res = {}
t0 = time.time()
cfgs = {
 'A_d140_r1200'    : dict(ds_tr=D10, decay=140.0, rounds=1200, alpha=0.5),
 'B_nodecay'       : dict(ds_tr=D10, decay=0.0,  rounds=1200, alpha=0.5),
 'C_excl_last_d140': dict(ds_tr=D9,  decay=140.0, rounds=1200, alpha=0.5),
 'D_d277'          : dict(ds_tr=D10, decay=277.0,rounds=1200, alpha=0.5),
 'E_r2400'         : dict(ds_tr=D10, decay=140.0, rounds=2400, alpha=0.5),
 'F_a052'          : dict(ds_tr=D10, decay=140.0, rounds=1200, alpha=0.52),
}
seeds = [7,8]
for name, kw in cfgs.items():
    p, ip = run(seeds=seeds, ds_pred=PV, **kw)
    base = base.merge(ip.assign(**{name:p}), on=['household_key','snapshot_day'], how='left')
    pv = base[name].values
    per = [np.abs(pv[base.snapshot_day==d]-yv[base.snapshot_day==d]).mean() for d in PV]
    res[name] = pv
    print('%-16s MAE %.3f  [375 %.2f | 403 %.2f | 431 %.2f]  bias %+.2f' % (
        name, np.abs(pv-yv).mean(), *per, (pv-yv).mean()))
print('elapsed %.0fs' % (time.time()-t0))

M = pd.DataFrame(res)
print('\nequal-weight blend of all 6: MAE %.3f' % np.abs(M.mean(axis=1).values-yv).mean())
print('blend of A+C+E: %.3f | A+B+D: %.3f | A+C: %.3f' % (
    np.abs(M[['A_d140_r1200','C_excl_last_d140','E_r2400']].mean(axis=1).values-yv).mean(),
    np.abs(M[['A_d140_r1200','B_nodecay','D_d277']].mean(axis=1).values-yv).mean(),
    np.abs(M[['A_d140_r1200','C_excl_last_d140']].mean(axis=1).values-yv).mean()))
print('\npairwise corr:')
print(M.corr().round(4))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings('ignore')

allF = load_saved('allF.parquet')
FEATS = [c for c in allF.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
X_all = allF[FEATS].astype(np.float64).replace([np.inf,-np.inf], np.nan)
tt = train_targets().rename(columns={'future_spend_4w':'y'})
y_map = tt.set_index(['household_key','snapshot_day'])['y']

def make(ds):
    m = allF.snapshot_day.isin(ds).values
    idx = allF.loc[m, ['household_key','snapshot_day']]
    return X_all[m], y_map.reindex(pd.MultiIndex.from_frame(idx)).values, idx

D_ALL = list(range(95,432,28))          # all 13 train snapshots
VAL = [459,487,515,543]
Xtr, ytr, itr = make(D_ALL)
print('train rows %d, feats %d, ymean %.1f' % (len(Xtr), len(FEATS), np.nanmean(ytr)))

Xp, _, ip = make(VAL)
SEEDS = [7,8,9,10,11,12,13,14]
t0 = time.time()
P = np.zeros((len(Xp), len(SEEDS)))
for k, s in enumerate(SEEDS):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
                         n_estimators=1200, learning_rate=0.03, max_depth=6,
                         min_child_weight=25, subsample=0.7, tree_method='hist',
                         n_jobs=16, random_state=s)
    m.fit(Xtr, ytr)                      # no time decay (pseudo-val winner)
    P[:, k] = np.clip(m.predict(Xp), 0, None)
    print('seed %d done %.0fs' % (s, time.time()-t0), flush=True)
pnew = P.mean(axis=1)

new = ip.copy(); new['prediction'] = pnew
e19 = load_saved('e019_blend_preds.parquet')
fin = e19[['household_key','snapshot_day']].merge(new, on=['household_key','snapshot_day'], how='left')
assert len(fin)==9989 and fin.prediction.notna().all()
fin['prediction'] = 0.5*fin.prediction + 0.5*e19.prediction.values
print('new-model stats: mean %.2f std %.2f min %.2f max %.2f' % (pnew.mean(), pnew.std(), pnew.min(), pnew.max()))
print('vs e019: corr %.4f  mean|diff| %.2f' % (np.corrcoef(fin.prediction, e19.prediction)[0,1],
      np.abs(fin.prediction-e19.prediction).mean()))
print('final: mean %.2f  rows %d  days %s' % (fin.prediction.mean(), len(fin), sorted(fin.snapshot_day.unique())))
path = save_table(fin[['household_key','snapshot_day','prediction']], 'e020_preds.parquet')
print('saved:', path)
