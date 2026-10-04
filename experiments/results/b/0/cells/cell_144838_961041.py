import numpy as np, pandas as pd

season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
print('NaN counts (nonzero only):')
nc = m.isna().sum()
print(nc[nc>0])

y = m.future_spend_4w.values
def mae(pred):
    p = np.nan_to_num(np.asarray(pred,float), nan=0.0)
    return round(np.mean(np.abs(y-p)),3)
print('\nfilled MAE own_ly:', mae(m.own_ly_spend4w), ' own_ly2:', mae(m.own_ly2_spend4w), ' season_lift:', mae(m.season_lift))
print('filled MAE spend56:', mae(m.spend56), ' spend112:', mae(m.spend112))

# local eval harness: ridge fit on snapshots 95..403, eval on 431 (pseudo-validation)
def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), verbose=True):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    # log1p the heavy-tailed spend-like columns
    logc = [i for i,c in enumerate(num) if ('spend' in c or 'disc' in c or 'lt_' in c or 'own_ly' in c)]
    X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))]) if logc else X
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed
        te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = df.future_spend_4w.values
        if target=='log': yt = np.log1p(yt)
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@yt[tr])
        p = Xs[te]@w
        if target=='log': p = np.expm1(p)
        res[ed] = round(np.mean(np.abs(yt[te]-p)),3)
    if verbose: print('local ridge MAE', res, 'n_feat', X.shape[1])
    return res

r = local_eval(m)
# baselines on 431
te = m.snapshot_day==431; yte = m.future_spend_4w[te].values
for c in ['spend28','spend56','own_ly_spend4w','own_ly2_spend4w','season_lift']:
    print('431 MAE', c, round(np.mean(np.abs(yte - m[c][te].fillna(0).values)),3))
print('431 MAE blend .5/.5:', round(np.mean(np.abs(yte - (0.5*m.spend28[te]+0.5*m.spend56[te]).fillna(0).values)),3))
print('431 median pred:', round(np.mean(np.abs(yte - np.median(m.future_spend_4w[m.snapshot_day<431]))),3))

v = agent_api.snapshot()
dep = v.products.department.value_counts()
print('\ndepartments:', len(dep))
print(dep.head(20))