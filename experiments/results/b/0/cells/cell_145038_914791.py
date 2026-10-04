import numpy as np, pandas as pd
season = agent_api.load_saved('season.parquet')
tt = agent_api.train_targets()
m = tt.merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def local_eval(df, lam=50.0, target='raw', eval_days=(403,431), logpat=('spend','disc','lt_','own_ly','red'), add_log=True):
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
        ymu = yt[tr].mean()
        A = Xs[tr].T@Xs[tr] + lam*np.eye(Xs.shape[1])
        w = np.linalg.solve(A, Xs[tr].T@(yt[tr]-ymu))
        p = Xs[te]@w + ymu
        if target=='log': p = np.expm1(np.clip(p,0,8))
        res[ed] = round(float(np.mean(np.abs(y[te]-p))),3)
    return res, X.shape[1]

subs = {
 'core_spend': ['spend28','spend56','spend112','spend364','lt_spend'],
 'rfm_all':    [c for c in m.columns if c.startswith(('spend','trips','actdays','qty','nprod','nstore'))],
 'trends':     [c for c in m.columns if c.startswith('trend')],
 'disc':       ['coupon_disc112','retail_disc112','coupon_match_disc112','disc_share112'],
 'timing':     ['recency','tenure','mean_hour112','mean_dow112'],
 'season':     ['own_ly_spend4w','season_lift','week_sin','week_cos'],
 'demo':       [c for c in m.columns if c.startswith('demo_')],
 'mkt':        [c for c in m.columns if c.startswith(('n_tgt','tA','tB','tC','days_since','targeted','red'))],
}
for name, cols in subs.items():
    cols = [c for c in cols if c in m.columns]
    r, nf = local_eval(m[['household_key','snapshot_day']+cols], lam=20)
    print(f'{name:12s}', r, nf)