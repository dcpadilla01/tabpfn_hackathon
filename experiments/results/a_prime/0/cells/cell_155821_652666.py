import agent_api, pandas as pd, numpy as np, numpy.linalg as la
v = agent_api.snapshot(459); tx = v.transactions
e = agent_api.load_saved('e011_rank.parquet'); t = agent_api.train_targets()
m = e.merge(t, on=['household_key','snapshot_day']).dropna(subset=['future_spend_4w'])
print('train rows', len(m))
def part_corr(df, feat, base):
    X = np.c_[np.ones(len(df)), base]
    y = np.log1p(df.future_spend_4w.values)
    b = la.lstsq(X,y,rcond=None)[0]; r = y - X@b
    f = df[feat].fillna(0).values
    return round(np.corrcoef(f, r)[0,1],4)
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-28)&(tx.day<=sd)]
    b = d.groupby(['household_key','basket_id'])['sales_value'].sum().reset_index()
    g = b.groupby('household_key')['sales_value']
    f = pd.DataFrame({'tot':g.sum(),'maxb':g.max(),'cnt':g.size()})
    f['max_share']=f.maxb/f.tot
    f['snapshot_day']=sd
    rows.append(f.reset_index())
F = pd.concat(rows); m = m.merge(F, on=['household_key','snapshot_day'], how='left')
m['max_share']=m.max_share.fillna(0)
lt = np.log1p(m.tot.fillna(0).values)
print('corr max_share w log-target:', round(np.corrcoef(m.max_share, np.log1p(m.future_spend_4w))[0,1],4))
print('partial corr max_share | log tot:', part_corr(m,'max_share', lt))
rows=[]
for sd in sorted(m.snapshot_day.unique()):
    d = tx[(tx.day>sd-84)&(tx.day<=sd)].copy()
    d['wk'] = (d.day+8)//7
    cur = (sd+8)//7
    s = d.groupby(['household_key','wk'])['sales_value'].sum().reset_index()
    p = s.pivot(index='household_key', columns='wk', values='sales_value')
    out = {}
    for k in range(1,13):
        col = cur-k
        out[f'w{k}'] = p[col] if col in p.columns else pd.Series(0.0, index=p.index)
    o = pd.DataFrame(out); o['snapshot_day']=sd
    rows.append(o.reset_index())
W = pd.concat(rows)
m = m.merge(W, on=['household_key','snapshot_day'], how='left')
wkcols=[f'w{k}' for k in range(1,13)]
m[wkcols]=m[wkcols].fillna(0)
m['w1_share'] = m.w1/(m[wkcols].sum(axis=1)+1)
m['w1_3_share'] = m[['w1','w2','w3']].sum(axis=1)/(m[wkcols].sum(axis=1)+1)
m['w4_12_share'] = m[[f'w{k}' for k in range(4,13)]].sum(axis=1)/(m[wkcols].sum(axis=1)+1)
lt84 = np.log1p(m[wkcols].sum(axis=1).values)
for c in ['w1_share','w1_3_share','w4_12_share']:
    print(c, 'partial | log84:', part_corr(m, c, lt84))
print(m[['w1_share','w1_3_share','w4_12_share']].describe().to_string())
# check e011 already has x_spend_p1 (prior aligned 4w block) — correlation of target with p1
print('corr log-target vs log p1:', part_corr(m,'x_spend_p1', np.zeros(len(m))))
