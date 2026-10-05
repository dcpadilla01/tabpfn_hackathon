
import pandas as pd, numpy as np

# quick look at existing feature columns to avoid duplication
for name in ['e007_te','e001_rfm','e002_composition']:
    try:
        df = agent_api.load_saved(name + '.parquet')
        print(name, df.shape)
        print([c for c in df.columns][:60])
        print()
    except Exception as ex:
        print(name, 'ERR', ex)

def make(view, snapshot_day):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    tx = tx[tx['household_key'].isin(hh)]
    piv = tx.pivot_table(index='household_key', columns='day', values='sales_value', aggfunc='sum')
    piv = piv.reindex(index=hh, columns=range(1, snapshot_day+1)).fillna(0.0)
    V = piv.values.astype(float)
    n = V.shape[0]
    csum = np.concatenate([np.zeros((n,1)), np.cumsum(V, axis=1)], axis=1)
    out = pd.DataFrame(index=hh)

    # rolling 28d spend windows sampled every 7 days back through history
    ends = list(range(snapshot_day, 27, -7))
    w28 = np.stack([csum[:, e] - csum[:, e-28] for e in ends], axis=1)
    m28 = w28.mean(axis=1); s28 = w28.std(axis=1)
    out['sp_w28_mean'] = m28
    out['sp_w28_std']  = s28
    out['sp_w28_min']  = w28.min(axis=1)
    out['sp_w28_max']  = w28.max(axis=1)
    out['sp_w28_med']  = np.median(w28, axis=1)
    out['sp_w28_cv']   = np.where(m28 > 0, s28/np.maximum(m28,1e-9), 0.0)
    out['sp_w28_zeroshare'] = (w28 < 0.01).mean(axis=1)

    # weekly (7d) spend process
    ends7 = list(range(snapshot_day, 6, -7))
    w7 = np.stack([csum[:, e] - csum[:, e-7] for e in ends7], axis=1)
    out['sp_w7_mean'] = w7.mean(axis=1)
    out['sp_w7_std']  = w7.std(axis=1)
    out['sp_w7_max']  = w7.max(axis=1)
    out['sp_w7_zeroshare'] = (w7 < 0.01).mean(axis=1)
    k = min(13, w7.shape[1])
    out['sp_w7_slope13'] = np.polyfit(np.arange(k), w7[:, :k].T, 1)[0] if k >= 2 else 0.0

    # current position vs own norm
    recent28 = csum[:, snapshot_day] - csum[:, snapshot_day-28]
    out['sp_recent_vs_typ']    = np.clip(recent28/(m28+1.0), 0, 5)
    out['sp_recent_minus_typ'] = recent28 - m28

    # trip cadence
    bt = tx[['household_key','basket_id','day']].drop_duplicates()
    bdays = bt.groupby('household_key')['day'].apply(lambda s: np.sort(s.unique()))
    rows = []
    for h in hh:
        d = bdays.get(h, None)
        if d is None or len(d) == 0:
            rows.append((np.nan,)*8 + (0,)); continue
        dsl = snapshot_day - d[-1]
        ia = np.diff(d)
        d112 = d[d > snapshot_day-112]; i112 = np.diff(d112)
        mi = ia.mean() if len(ia) else np.nan
        mic = np.clip(mi,1,56) if mi == mi else np.nan
        rows.append((mi,
                     np.median(ia) if len(ia) else np.nan,
                     ia.std() if len(ia) else np.nan,
                     i112.mean() if len(i112) else np.nan,
                     np.median(i112) if len(i112) else np.nan,
                     dsl,
                     28.0/mic if mi == mi else np.nan,
                     dsl/mic if (mi == mi) else np.nan,
                     int((d > snapshot_day-28).sum())))
    cadf = pd.DataFrame(rows, index=hh,
        columns=['cad_int_mean','cad_int_med','cad_int_std','cad_int_mean_112','cad_int_med_112','cad_dsl','cad_exp_trips28','cad_phase','cad_trips28'])
    out = out.join(cadf)
    if snapshot_day == 95:
        print('sample snapshot 95:', out.shape); print(out.head(3).T)
    return out

nf = agent_api.build_features(make)
print('nf shape', nf.shape)
if 'household_key' in nf.columns:
    nf2 = nf.copy()
else:
    nf2 = nf.reset_index()
    if 'index' in nf2.columns: nf2 = nf2.drop(columns=['index'])
try:
    base = agent_api.load_saved('e007_te.parquet')
except Exception:
    base = agent_api.load_saved('e007_te')
print('base', base.shape)
merged = base.merge(nf2, on=['household_key','snapshot_day'], how='inner')
print('merged', merged.shape, 'n_features', merged.shape[1]-2)
assert len(merged) == len(base), 'row count mismatch'
path = agent_api.save_table(merged, 'e008_spendproc.parquet')
print('saved', path)
