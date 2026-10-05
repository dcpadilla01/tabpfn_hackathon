import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def peer_feats(view, d, ks=(25,50,100)):
    tx = view.table('transactions')
    hh = HH[d]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    agg = cur.groupby('household_key').agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    agp = prv.groupby('household_key').agg(spp=('sales_value','sum'))
    first = tx.groupby('household_key').day.min()
    last = tx.groupby('household_key').day.max()
    idx = first.index
    rec = np.clip(d - last.reindex(idx).values, 0, None)
    ten = (d - first.values + 1).astype(float)
    sp_c = agg.sp.reindex(idx).fillna(0).values
    tr_c = agg.tr.reindex(idx).fillna(0).values
    sp_p = agp.spp.reindex(idx).fillna(0).values
    X = np.column_stack([np.log1p(sp_c), np.log1p(tr_c), np.log1p(rec), np.log1p(ten), np.log1p(sp_p)])
    ref_idx = np.where(first.values <= d-112)[0]
    q_idx = idx.get_indexer(hh)
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    out = {}
    for k in ks:
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    return pd.DataFrame(out, index=hh)

t0=time.time()
v = api.snapshot(as_of_day=431)
pf = peer_feats(v, 431)
print('time:', round(time.time()-t0,1),'s'); print(pf.describe().round(2))

tt = api.train_targets()
e15 = api.load_saved('e015_stack.parquet')
s431 = e15[e15.snapshot_day==431].set_index('household_key')[['stack_ridge']]
mm = pf.join(s431, how='inner').join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
yy = mm.future_spend_4w
print('n:', len(mm), 'stack MAE 431:', round(np.abs(mm.stack_ridge-yy).mean(),2))
for c in pf.columns:
    print(c, 'MAE:', round(np.abs(mm[c]-yy).mean(),2))
for w in [0.3,0.5,0.7]:
    print(f'blend stack w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med50-yy).mean(),2))
