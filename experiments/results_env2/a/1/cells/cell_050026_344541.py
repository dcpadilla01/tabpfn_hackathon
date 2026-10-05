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
