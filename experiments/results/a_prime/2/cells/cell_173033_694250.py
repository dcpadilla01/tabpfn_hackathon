import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
drop_cols = [c for c in base.columns if c in ('tlag_13','index','wyoy','wyoy_ratio','yoy_spend364','yoy_ratio')]
v1 = base.drop(columns=drop_cols)

def fn(view, snapshot_day):
    s = snapshot_day
    keys = np.asarray(view.households)
    tx = view.transactions
    res = pd.DataFrame(index=keys)
    wk = (tx.day + 8)//7
    tx = tx.assign(wk=wk, seas=wk % 52)
    nxt = [(s + 8)//7 + 1 + k for k in range(4)]
    nxt_seas = [w % 52 for w in nxt]
    g = tx.groupby(['household_key','seas'])['sales_value'].sum()
    hh_tot = tx.groupby('household_key')['sales_value'].sum()
    prof = g.unstack('seas').reindex(columns=range(52))
    prof = prof.reindex(keys).fillna(0.0)
    nweeks_obs = tx.groupby('household_key')['wk'].nunique().reindex(keys).fillna(1.0)
    weekly_mean = (hh_tot.reindex(keys).fillna(0.0) / nweeks_obs).replace(0, np.nan)
    pmean = prof.mean(axis=1)
    rel = (prof.div(pmean.replace(0,np.nan), axis=0))
    seas_score = np.nanmean(rel[nxt_seas].values, axis=1)
    res['hh_season_next4'] = pd.Series(np.nan_to_num(seas_score), index=keys)
    res['hh_season_next4_x'] = pd.Series(np.nan_to_num(seas_score), index=keys) * pd.Series(weekly_mean.values, index=keys).fillna(0).values
    gseas = tx.groupby('seas')['sales_value'].sum()
    gm = gseas.mean()
    res['glob_season_next4'] = float(np.mean([gseas.get(ss, 0.0)/gm for ss in nxt_seas]))
    zr = np.zeros(len(keys)); wm = np.zeros(len(keys)); wsd = np.zeros(len(keys))
    hhmap = {h:i for i,h in enumerate(keys)}
    wins = []
    for k in range(12):
        hi, lo = s - 7*k, s - 28 - 7*k
        if lo < 1: break
        wins.append((hi,lo))
    if wins:
        m = tx[((tx.day <= wins[0][0]) & (tx.day > wins[-1][1]))]
        m = m[m.household_key.isin(hhmap)]
        m = m.assign(hi=m.household_key.map(hhmap))
        arr = np.zeros((len(keys), len(wins)))
        for j,(hi,lo) in enumerate(wins):
            sub = m[(m.day <= hi) & (m.day > lo)]
            agg = sub.groupby('hi')['sales_value'].sum()
            arr[agg.index.values, j] = agg.values
        zr = (arr == 0).mean(axis=1)
        wm = arr.mean(axis=1); wsd = arr.std(axis=1)
    res['zrate_7s'] = zr
    res['wmean_7s'] = wm
    res['wcv_7s'] = np.where(wm>0, wsd/np.maximum(wm,1e-9), 0.0)
    last = tx.groupby('household_key')['day'].max().reindex(keys)
    gaps = tx.sort_values('day').groupby('household_key')['day'].apply(lambda x: np.diff(np.unique(x.values)).mean() if len(np.unique(x))>1 else np.nan)
    med_gap = gaps.reindex(keys)
    res['due_ratio'] = ((last.reindex(keys).fillna(s) - s).abs() / med_gap.replace(0,np.nan)).replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return res

new = agent_api.build_features(fn)
print('new', new.shape, list(new.columns))
m2 = base.merge(new, on=['household_key','snapshot_day'], how='inner')
print('m2', m2.shape)

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
r1 = ev(v1); print('v1 drop-shift: %.2f / %.2f  d_inner %+.2f'%(r1[0],r1[1],r1[1]-b[1]))
r2 = ev(m2); print('v2 +season/hazard: %.2f / %.2f  d_inner %+.2f'%(r2[0],r2[1],r2[1]-b[1]))
nc = [c for c in new.columns if c not in ('household_key','snapshot_day')]
df2 = m2.merge(tt, on=['household_key','snapshot_day'])
print(df2[nc].corrwith(df2.future_spend_4w))
iv2 = df2[df2.snapshot_day>=375]; tr2 = df2[df2.snapshot_day<=347]
for c in nc:
    print('  %-18s tr %8.2f iv %8.2f' % (c, tr2[c].mean(), iv2[c].mean()))
p = agent_api.save_table(m2, 'e017_season_hazard.parquet'); print('saved', p)