
import pandas as pd, numpy as np

def make(view, snapshot_day):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    tx = tx[tx['household_key'].isin(hh)]
    g = tx.groupby(['household_key','day'])['sales_value'].sum()
    g = g.reset_index()
    out = pd.DataFrame(index=hh)

    # window sums via searchsorted on per-household sorted day arrays
    ends28 = list(range(snapshot_day, 27, -7))          # 28d windows ending every 7d
    ends7  = list(range(snapshot_day, 6, -7))           # 7d windows ending every 7d
    rows = {c: np.full(len(hh), np.nan) for c in
            ['sp_w28_mean','sp_w28_std','sp_w28_min','sp_w28_max','sp_w28_med','sp_w28_cv',
             'sp_w28_zeroshare','sp_w7_mean','sp_w7_std','sp_w7_max','sp_w7_zeroshare',
             'sp_w7_slope13','sp_recent_vs_typ','sp_recent_minus_typ',
             'cad_int_mean','cad_int_med','cad_int_std','cad_int_mean_112','cad_int_med_112',
             'cad_dsl','cad_exp_trips28','cad_phase','cad_trips28']}
    bdays = tx[['household_key','basket_id','day']].drop_duplicates()
    bmap = {h: np.sort(v) for h, v in bdays.groupby('household_key')['day']}
    gmap = {h: a for h, a in g.groupby('household_key')[['day','sales_value']]}

    for i, h in enumerate(hh):
        sub = gmap.get(h)
        if sub is None or len(sub) == 0:
            continue
        d = sub['day'].values.astype(np.int64)
        s = sub['sales_value'].values.astype(float)
        cs = np.concatenate([[0.0], np.cumsum(s)])
        def wsum(e, w):
            hi = np.searchsorted(d, e, side='right')
            lo = np.searchsorted(d, e - w, side='right')
            return cs[hi] - cs[lo]
        w28 = np.array([wsum(e, 28) for e in ends28])
        w7  = np.array([wsum(e, 7)  for e in ends7])
        m28 = w28.mean(); s28 = w28.std()
        rows['sp_w28_mean'][i]=m28; rows['sp_w28_std'][i]=s28
        rows['sp_w28_min'][i]=w28.min(); rows['sp_w28_max'][i]=w28.max()
        rows['sp_w28_med'][i]=np.median(w28)
        rows['sp_w28_cv'][i]= s28/max(m28,1e-9) if m28>0 else 0.0
        rows['sp_w28_zeroshare'][i]=(w28<0.01).mean()
        rows['sp_w7_mean'][i]=w7.mean(); rows['sp_w7_std'][i]=w7.std()
        rows['sp_w7_max'][i]=w7.max(); rows['sp_w7_zeroshare'][i]=(w7<0.01).mean()
        k=min(13,len(w7))
        rows['sp_w7_slope13'][i]=np.polyfit(np.arange(k), w7[:k], 1)[0]
        r28 = wsum(snapshot_day,28)
        rows['sp_recent_vs_typ'][i]=np.clip(r28/(m28+1.0),0,5)
        rows['sp_recent_minus_typ'][i]=r28-m28
        bd = bmap.get(h)
        if bd is not None and len(bd):
            dsl = snapshot_day - bd[-1]
            ia = np.diff(bd)
            d112 = bd[bd > snapshot_day-112]; i112 = np.diff(d112)
            mi = ia.mean() if len(ia) else np.nan
            mic = min(max(mi,1),56) if mi==mi else np.nan
            rows['cad_int_mean'][i]=mi
            rows['cad_int_med'][i]=np.median(ia) if len(ia) else np.nan
            rows['cad_int_std'][i]=ia.std() if len(ia) else np.nan
            rows['cad_int_mean_112'][i]=i112.mean() if len(i112) else np.nan
            rows['cad_int_med_112'][i]=np.median(i112) if len(i112) else np.nan
            rows['cad_dsl'][i]=dsl
            rows['cad_exp_trips28'][i]=28.0/mic if mi==mi else np.nan
            rows['cad_phase'][i]=dsl/mic if mi==mi else np.nan
            rows['cad_trips28'][i]=(bd>snapshot_day-28).sum()
    out = pd.DataFrame(rows, index=hh)
    return out

nf = agent_api.build_features(make)
nf2 = nf.reset_index()
if 'index' in nf2.columns: nf2 = nf2.drop(columns=['index'])
print('nf shape', nf.shape, 'cols ok:', nf2.columns.tolist()[:5])
base = agent_api.load_saved('e007_te.parquet')
print('base', base.shape)
merged = base.merge(nf2, on=['household_key','snapshot_day'], how='inner')
print('merged', merged.shape, 'new feats', merged.shape[1]-base.shape[1])
assert len(merged)==len(base)
path = agent_api.save_table(merged, 'e008_spendproc.parquet')
print('saved', path)
