import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.table('transactions')
    hh = HH[d]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    g = tx.groupby('household_key').day
    first, last = g.min(), g.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    ref_idx = np.where(first.values <= d-84)[0]
    pos_of_hh = {k: p for p, k in enumerate(idx[ref_idx])}
    q_idx = idx.get_indexer(hh)
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    # exclude self from reference pool
    for i, k in enumerate(np.asarray(hh)):
        p = pos_of_hh.get(k)
        if p is not None: d2[i, p] = np.inf
    out = {}
    for k in (25, 50):
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    return pd.DataFrame(out, index=hh)

t0 = time.time()
tab = api.build_features(fn)
print('built:', tab.shape, 'time:', round(time.time()-t0,1))
print(tab.groupby('snapshot_day').knn_med50.mean().round(1))
p = api.save_table(tab, 'e019_peer')
print('saved:', p)
