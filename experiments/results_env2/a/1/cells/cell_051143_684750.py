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
