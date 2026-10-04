import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
m2 = agent_api.load_saved('e017_season_hazard.parquet')
tt = agent_api.train_targets()

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions
    res = pd.DataFrame(index=keys)
    hhmap = {h:i for i,h in enumerate(keys)}
    # stride-7 28d windows, last 16
    wins = [(s-7*k, s-28-7*k) for k in range(16) if s-28-7*k >= 1]
    arr = np.zeros((len(keys), len(wins)))
    if wins:
        m = tx[tx.household_key.isin(hhmap)].copy()
        m['hi'] = m.household_key.map(hhmap)
        for j,(hi,lo) in enumerate(wins):
            sub = m[(m.day <= hi) & (m.day > lo)]
            agg = sub.groupby('hi')['sales_value'].sum()
            arr[agg.index.values, j] = agg.values
    zr = (arr==0).mean(axis=1)
    wm = arr.mean(axis=1); wsd = arr.std(axis=1)
    res['zrate_7s'] = zr
    res['wmean_7s'] = wm
    res['wcv_7s'] = np.where(wm>0, wsd/np.maximum(wm,1e-9), 0.0)
    res['wmax_7s'] = arr.max(axis=1)
    res['zrate_recent'] = (arr[:,:6]==0).mean(axis=1)
    res['zrate_older'] = (arr[:,6:12]==0).mean(axis=1)
    res['zdiff'] = res['zrate_older'] - res['zrate_recent']
    # inter-trip gap stats (last 182d)
    t2 = tx[(tx.day > s-182) & (tx.day <= s)]
    g = t2.sort_values('day').groupby('household_key')['day'].apply(
        lambda x: np.diff(np.unique(x.values)) if len(np.unique(x))>1 else np.array([np.nan]))
    gm = g.apply(lambda a: np.nanmean(a)).reindex(keys).astype(float)
    gs = g.apply(lambda a: np.nanstd(a)).reindex(keys).astype(float)
    res['gap_mean'] = gm.fillna(84.0)
    res['gap_std'] = gs.fillna(0.0)
    res['gap_cv'] = (res['gap_std']/res['gap_mean'].replace(0,np.nan)).fillna(0.0)
    last = tx.groupby('household_key')['day'].max().reindex(keys).fillna(s)
    res['due_ratio'] = ((last - s).abs()/res['gap_mean'].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    res['p_active_x_wmean'] = (1-zr)*wm
    return res

new = agent_api.build_features(fn)
print('new', new.shape, [c for c in new.columns if c not in ('household_key','snapshot_day')])
m3 = base.merge(new, on=['household_key','snapshot_day'], how='inner')

def ev(df, alpha=100.0):
    df = df.merge(tt, on=['household_key','snapshot_day'], how='inner')
    tr = df[df.snapshot_day <= 347]; iv = df[df.snapshot_day >= 375]
    cols=[c for c in df.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    A = tr[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    B = iv[cols].apply(pd.to_numeric, errors='coerce').astype(float)
    mu=A.mean(); sd=A.std().replace(0,1)
    A=((A-mu)/sd).fillna(0).values; B=((B-mu)/sd).fillna(0).values
    ytr=tr.future_spend_4w.values; yiv=iv.future_spend_4w.values
    w=np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(B@w-yiv).mean()

b = ev(base); print('E009 base: %.2f / %.2f'%b)
r2 = ev(m2); print('v2 season/hazard: %.2f / %.2f  d %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
r3 = ev(m3); print('v3 extended:      %.2f / %.2f  d %+.2f'%(r3[0],r3[1],r3[1]-b[1]))
# v4: v3 + interactions with ewma/spend
m4 = m3.copy()
m4['ewma4_x_pact'] = m4['ewma_4']*(1-m4['zrate_7s'])
m4['spend28_x_wcv'] = m4['spend_28']*m4['wcv_7s']
m4['ewma4_x_due'] = m4['ewma_4']*m4['due_ratio']
r4 = ev(m4); print('v4 +interactions: %.2f / %.2f  d %+.2f'%(r4[0],r4[1],r4[1]-b[1]))
p = agent_api.save_table(m4, 'e017_hazard_ext.parquet'); print('saved', p)