import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly'), add_log=True):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if add_log and logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = y if target=='raw' else np.log1p(y)
        # center y with train mean -> intercept
        ymu = yt[tr].mean()
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@(yt[tr]-ymu))
        p = Xs[te]@w + ymu
        if target=='log': p = np.expm1(np.clip(p,0,8))
        res[ed] = round(float(np.mean(np.abs(y[te]-p))),3)
    return res, X.shape[1]

print('season w/ intercept:', local_eval(m))
print('season log-target  :', local_eval(m, target='log'))
print('season raw no-logX :', local_eval(m, add_log=False))
print('spend28 only       :', local_eval(m[['household_key','snapshot_day','spend28','future_spend_4w']], lam=10))
print('spend28+56+112     :', local_eval(m[['household_key','snapshot_day','spend28','spend56','spend112','future_spend_4w']], lam=10))
print('spend28 log-target :', local_eval(m[['household_key','snapshot_day','spend28','future_spend_4w']], lam=10, target='log'))
print('spend28+56 log-tgt :', local_eval(m[['household_key','snapshot_day','spend28','spend56','future_spend_4w']], lam=10, target='log'))

# power-law single feature: fit target = a*spend28^b via log-log on train(<431)
tr = m.snapshot_day<431; te = m.snapshot_day==431
mask = m.spend28[tr]>0
b, la = np.polyfit(np.log(m.spend28[tr][mask]), np.log1p(y[tr][mask]), 1)
p = np.expm1(la + b*np.log(np.clip(m.spend28[te],1e-9,None)))
print('power-law spend28 431 MAE:', round(float(np.mean(np.abs(y[te]-p))),3), 'b=',round(b,3))

# ratio target/spend28 by bucket
r = y/np.clip(m.spend28,1,None)
bins = pd.qcut(m.spend28, 8, duplicates='drop')
print('\nmedian ratio by spend28 bucket:')
print(m.groupby(bins, observed=True).apply(lambda g: pd.Series({'med_ratio': np.median(g.future_spend_4w/np.clip(g.spend28,1,None)), 'n':len(g)}), include_groups=False).round(2))