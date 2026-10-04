import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values

# which snapshot days have own_ly_spend4w non-NaN?
print('own_ly non-NaN count by snapshot day:')
print(m.groupby('snapshot_day').own_ly_spend4w.agg(['count','mean']).round(1))

# pseudo-validation: ridge on 95..403, eval 431, using season table (no own_ly)
def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly')):
    num = [c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w') and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num].replace([np.inf,-np.inf], np.nan).fillna(0).values.astype(float)
    logc = [i for i,c in enumerate(num) if any(p in c for p in logpat)]
    if logc: X = np.hstack([X, np.log1p(np.clip(X[:,logc],0,None))])
    res = {}
    for ed in eval_days:
        tr = df.snapshot_day < ed; te = df.snapshot_day == ed
        mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
        Xs = (X-mu)/sd
        yt = df.future_spend_4w.values
        if target=='log': yt = np.log1p(yt)
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@yt[tr])
        p = Xs[te]@w
        if target=='log': p = np.expm1(p)
        res[ed] = round(float(np.mean(np.abs(yt[te]-p))),3)
    return res, X.shape[1]

r, nf = local_eval(m)
print('season table local ridge:', r, 'nfeat', nf)
rich = agent_api.load_saved('rich_behavioral.parquet')
mr = tt.merge(rich, on=['household_key','snapshot_day'], how='left')
r2, nf2 = local_eval(mr)
print('rich_behavioral local ridge:', r2, 'nfeat', nf2)
# raw spend28-only "model": local
for lam in [10,50,200]:
    X = m[['spend28']].values.astype(float)
    tr = m.snapshot_day<431; te = m.snapshot_day==431
    mu,sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Xs=(X-mu)/sd
    A=Xs[tr].T@Xs[tr]+lam*np.eye(1); w=np.linalg.solve(A,Xs[tr].T@y[tr])
    print('ridge spend28 lam',lam,'431 MAE', round(float(np.mean(np.abs(y[te]-Xs[te]*w))),3))