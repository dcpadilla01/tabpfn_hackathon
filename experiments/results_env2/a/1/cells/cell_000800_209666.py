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
