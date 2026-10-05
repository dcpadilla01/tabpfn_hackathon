import numpy as np, pandas as pd
f4 = agent_api.load_saved('e004_features.parquet')
f5 = agent_api.load_saved('e005_newfeats.parquet')
p5 = agent_api.load_saved('e005_preds.parquet')
tt = agent_api.train_targets()
print('f4', f4.shape, 'f5', f5.shape, 'p5', p5.shape, 'tt', tt.shape)
print('f4 cols:', list(f4.columns))
print('f5 cols:', list(f5.columns))
print('f4 days:', sorted(f4.snapshot_day.unique()))
print('p5 days:', sorted(p5.snapshot_day.unique()))
y = tt.future_spend_4w
print(y.describe())
print('zero frac %.3f  median %.1f  mean %.1f' % ((y==0).mean(), y.median(), y.mean()))
print('pct', np.percentile(y, [50,75,90,95,99]))
c4, c5 = set(f4.columns), set(f5.columns)
print('overlap cols:', c4 & c5)
print(f4.dtypes.value_counts())


# ---- cell ----
import numpy as np, pandas as pd
f4 = agent_api.load_saved('e004_features.parquet')
f5 = agent_api.load_saved('e005_newfeats.parquet')
F = f4.merge(f5, on=['household_key','snapshot_day'], how='left')
tt = agent_api.train_targets()
F = F.merge(tt, on=['household_key','snapshot_day'], how='left')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
print('merged', F.shape, 'n feats', len(FEATS))
path = agent_api.save_table(F, 'allF.parquet')
print(path)


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
p5 = agent_api.load_saved('e005_preds.parquet')
# E005 per-day MAE on validation
tt = agent_api.train_targets()
m = p5.merge(tt, on=['household_key','snapshot_day'], how='left')
for d,g in m.groupby('snapshot_day'):
    print(d, 'MAE %.3f' % np.abs(g.prediction-g.future_spend_4w).mean(), 'n=%d'%len(g),
          'pred_mean %.1f true_mean %.1f' % (g.prediction.mean(), g.future_spend_4w.mean()))

def decay_w(days, half=140):
    return 0.5 ** ((days.max() - days) / half)

def run_xgb(tr, va, feats, lr=0.03, rounds=2400, half=140, alpha=0.5, depth=0, leaves=31,
            subs=0.8, cols=0.8, mincw=20, seed=7, early=None, verbose=False):
    Xtr, ytr = tr[feats].values, tr.future_spend_4w.values
    w = decay_w(tr.snapshot_day.astype(float).values, half)
    model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=rounds, learning_rate=lr, max_depth=depth, max_leaves=leaves,
        grow_policy='lossguide', subsample=subs, colsample_bytree=cols,
        min_child_weight=mincw, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    model.fit(Xtr, ytr, sample_weight=w, eval_set=[(va[feats].values, va.future_spend_4w.values)],
              verbose=False)
    return model, model.predict(va[feats].values)

t0=time.time()
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]
mod, pred = run_xgb(tr, va, FEATS)
print('internal 431 MAE: %.3f  (%.0fs)' % (np.abs(pred-va.future_spend_4w.values).mean(), time.time()-t0))
# baseline: predict train median
print('median pred MAE: %.3f' % np.abs(np.full(len(va), tr.future_spend_4w.median())-va.future_spend_4w.values).mean())


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]

def fit_predict(lr, rounds, half, leaves=31, subs=0.8, cols=0.8, mincw=20, seed=7):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=leaves,
        grow_policy='lossguide', subsample=subs, colsample_bytree=cols,
        min_child_weight=mincw, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[FEATS].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[FEATS].values)

t0=time.time()
cfgs = [
    dict(lr=0.03, rounds=2400, half=140),          # E005 config (reference)
    dict(lr=0.03, rounds=2400, half=277),          # E003 decay
    dict(lr=0.05, rounds=1500, half=140),
    dict(lr=0.03, rounds=2400, half=140, cols=0.6),
    dict(lr=0.03, rounds=2400, half=140, mincw=50),
]
for c in cfgs:
    p = fit_predict(**c)
    print(c, 'MAE %.3f' % np.abs(p-va.future_spend_4w.values).mean(), ' (%.0fs)'%(time.time()-t0))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()]
va = F[F.snapshot_day==431]
yv = va.future_spend_4w.values
print('target mean by train day:')
print(tr.groupby('snapshot_day').future_spend_4w.agg(['mean','median']).round(1).T)

def fit_predict(feats, lr=0.03, rounds=2400, half=140, alpha=0.5, seed=7, leaves=31):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=leaves,
        grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[feats].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[feats].values)

t0=time.time()
# A: day-index feature
trA = tr.assign(day_idx=tr.snapshot_day.astype(float)); vaA = va.assign(day_idx=va.snapshot_day.astype(float))
pA = fit_predict(FEATS+['day_idx'])
print('A day_idx:      MAE %.3f (%.0fs)' % (np.abs(pA-yv).mean(), time.time()-t0))

# B: 8-seed ensemble (ref feats)
preds = [fit_predict(FEATS, seed=s) for s in range(8)]
pB = np.mean(preds, axis=0)
print('B 8-seed:       MAE %.3f (%.0fs)' % (np.abs(pB-yv).mean(), time.time()-t0))

# C: alpha blend
pA5 = preds[0]
for a in (0.45, 0.55):
    pa = fit_predict(FEATS, alpha=a, seed=0)
    blend = a*pA5/0.5 + (0.5-a)/0.5*pa
    print('C alpha %.2f blend: MAE %.3f' % (a, np.abs(blend-yv).mean()), '(%.0fs)'%(time.time()-t0))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()].copy()
va = F[F.snapshot_day==431].copy()
yv = va.future_spend_4w.values

def fit_predict(trF, vaF, feats, lr=0.03, rounds=2400, half=140, alpha=0.5, seed=7):
    w = 0.5 ** ((trF.snapshot_day.max() - trF.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=31,
        grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(trF[feats].values, trF.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(vaF[feats].values)

t0=time.time()
# A: day-index feature
trA = tr.assign(day_idx=tr.snapshot_day.astype(float)); vaA = va.assign(day_idx=va.snapshot_day.astype(float))
pA = fit_predict(trA, vaA, FEATS+['day_idx'])
print('A day_idx:      MAE %.3f (%.0fs)' % (np.abs(pA-yv).mean(), time.time()-t0))

# B: 8-seed ensemble (ref feats)
preds = [fit_predict(tr, va, FEATS, seed=s) for s in range(8)]
pB = np.mean(preds, axis=0)
print('B 8-seed:       MAE %.3f (%.0fs)' % (np.abs(pB-yv).mean(), time.time()-t0))

# C: alpha blend
pA5 = preds[0]
for a in (0.45, 0.55):
    pa = fit_predict(tr, va, FEATS, alpha=a, seed=0)
    blend = a*pA5/0.5 + (0.5-a)/0.5*pa
    print('C alpha %.2f blend: MAE %.3f' % (a, np.abs(blend-yv).mean()), '(%.0fs)'%(time.time()-t0))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
va = F[F.snapshot_day==431]
yv = va.future_spend_4w.values
# heuristics
for name, p in [('spend_28', va.spend_28.values), ('spend_84', va.spend_84.values),
                ('(28+84)/2', (va.spend_28.values+va.spend_84.values)/2),
                ('spend_56', va.spend_56.values)]:
    print('%-10s MAE %.3f' % (name, np.abs(p-yv).mean()))
print('E005-config xgb internal: 63.049 (ref)')

# build lag features via fresh build_features (only need lag windows; cheap fn)
def lag_fn(view, day):
    tx = view.table('transactions')
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    out = pd.DataFrame(index=hh)
    for lo, hi, nm in [(56,28,'lag28_56'), (84,56,'lag56_84'), (112,84,'lag84_112'), (140,112,'lag112_140')]:
        w = g[(g.day > day-hi) & (g.day <= day-lo)].groupby('household_key').sales_value.sum()
        out[nm] = w.reindex(hh).fillna(0.0)
    return out

t0=time.time()
L = agent_api.build_features(lag_fn)
print('lag feats built %.0fs' % (time.time()-t0), L.shape)
path = agent_api.save_table(L.reset_index(), 'lagfeats.parquet')
print(path)


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
L = agent_api.load_saved('lagfeats.parquet')
F = F.merge(L, on=['household_key','snapshot_day'], how='left')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()].copy()
va = F[F.snapshot_day==431].copy()
yv = va.future_spend_4w.values
LAG = ['lag28_56','lag56_84','lag84_112','lag112_140']
print('corr with target (431):', {c: round(np.corrcoef(va[c], yv)[0,1],3) for c in LAG})

def fit_predict(feats, lr=0.03, rounds=2400, half=140, seed=7):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5,
        n_estimators=rounds, learning_rate=lr, max_depth=0, max_leaves=31,
        grow_policy='lossguide', subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0, n_jobs=8, random_state=seed, tree_method='hist')
    m.fit(tr[feats].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[feats].values)

t0=time.time()
pL = fit_predict(FEATS)
print('with lag feats: MAE %.3f (%.0fs)' % (np.abs(pL-yv).mean(), time.time()-t0))
# blend lag model with reference 63.049? just report; also try 4-seed avg of lag model
preds=[fit_predict(FEATS, seed=s) for s in range(4)]
print('lag 4-seed:     MAE %.3f' % np.abs(np.mean(preds,axis=0)-yv).mean())


# ---- cell ----
import numpy as np, pandas as pd
L = agent_api.load_saved('lagfeats.parquet')
print(L.dtypes)
print(L.groupby('snapshot_day')[['lag28_56','lag56_84','lag84_112','lag112_140']].mean().round(2))
v = L[L.snapshot_day==431]
print('431 std:', v[['lag28_56','lag56_84','lag84_112','lag112_140']].std().values)
print(v.head())


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
# --- corrected lag features via build_features ---
def lag_fn(view, day):
    tx = view.table('transactions')
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    out = pd.DataFrame(index=hh)
    for lo, hi, nm in [(56,28,'lag28_56'), (84,56,'lag56_84'), (112,84,'lag84_112'), (140,112,'lag112_140')]:
        w = g[(g.day > day-lo) & (g.day <= day-hi)].groupby('household_key').sales_value.sum()
        out[nm] = w.reindex(hh).fillna(0.0)
    return out
t0=time.time()
L = agent_api.build_features(lag_fn)
print('lag built %.0fs' % (time.time()-t0))
print(L.groupby('snapshot_day')[['lag28_56','lag56_84']].mean().round(1).tail(6))
path = agent_api.save_table(L.reset_index(), 'lagfeats2.parquet')

# --- internal test ---
F = agent_api.load_saved('allF.parquet').merge(L.reset_index(), on=['household_key','snapshot_day'], how='left')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[(F.snapshot_day<=403) & F.future_spend_4w.notna()].copy()
va = F[F.snapshot_day==431].copy()
yv = va.future_spend_4w.values
print('corr lag28_56 vs y @431: %.3f' % np.corrcoef(va.lag28_56, yv)[0,1])
def fit_predict(feats, seed=7, half=140):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / half)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=2400,
        learning_rate=0.03, max_depth=0, max_leaves=31, grow_policy='lossguide', subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=8,
        random_state=seed, tree_method='hist')
    m.fit(tr[feats].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    return m.predict(va[feats].values)
t0=time.time()
p = fit_predict(FEATS)
print('lags + ref: MAE %.3f (%.0fs)' % (np.abs(p-yv).mean(), time.time()-t0))
p2 = fit_predict(FEATS+['day_idx'] if False else FEATS)


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[F.future_spend_4w.notna()]          # ALL train snapshots incl. 431
p5 = agent_api.load_saved('e005_preds.parquet')
va = F[F.snapshot_day>=459]
va = p5[['household_key','snapshot_day']].merge(va, on=['household_key','snapshot_day'], how='left')
assert len(va)==len(p5) and va[FEATS].notna().all().all()

t0=time.time()
preds=[]
for s in (7, 21, 99):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / 140)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=2400,
        learning_rate=0.03, max_depth=0, max_leaves=31, grow_policy='lossguide', subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=8,
        random_state=s, tree_method='hist')
    m.fit(tr[FEATS].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    preds.append(m.predict(va[FEATS].values))
    print('seed', s, 'done (%.0fs)' % (time.time()-t0))
va = va.assign(prediction=np.mean(preds, axis=0))
out = va[['household_key','snapshot_day','prediction']].rename(columns={'prediction':'prediction'})
print(out.shape, out.prediction.describe().round(2).to_dict())
path = agent_api.save_table(out, 'e008_preds.parquet')
print('SAVED', path)


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time
F = agent_api.load_saved('allF.parquet')
FEATS = [c for c in F.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
tr = F[F.future_spend_4w.notna()]
p5 = agent_api.load_saved('e005_preds.parquet')
va = p5[['household_key','snapshot_day']].merge(F[F.snapshot_day>=459], on=['household_key','snapshot_day'], how='left')
print('rows', len(va), 'nan feats:', int(va[FEATS].isna().any(axis=1).sum()))
va[FEATS] = va[FEATS].fillna(0).replace([np.inf,-np.inf], 0)
t0=time.time()
preds=[]
for s in (7, 21, 99):
    w = 0.5 ** ((tr.snapshot_day.max() - tr.snapshot_day.astype(float).values) / 140)
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, n_estimators=2400,
        learning_rate=0.03, max_depth=0, max_leaves=31, grow_policy='lossguide', subsample=0.8,
        colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0, n_jobs=8,
        random_state=s, tree_method='hist')
    m.fit(tr[FEATS].values, tr.future_spend_4w.values, sample_weight=w, verbose=False)
    preds.append(m.predict(va[FEATS].values))
va = va.assign(prediction=np.mean(preds, axis=0))
out = va[['household_key','snapshot_day','prediction']]
print(out.shape, 'finite:', np.isfinite(out.prediction).all())
path = agent_api.save_table(out, 'e008_preds.parquet')
print('SAVED', path)
