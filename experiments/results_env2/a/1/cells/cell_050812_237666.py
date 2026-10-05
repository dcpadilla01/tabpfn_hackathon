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
