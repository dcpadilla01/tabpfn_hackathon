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
