import agent_api as api
import numpy as np, pandas as pd, time

base = api.baseline_features()
HH = {int(d): pd.Index(np.asarray(base[base.snapshot_day==d].household_key.unique())) for d in sorted(base.snapshot_day.unique())}

def peer_feats(view, d):
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
    # ref pool: tenure >= 84d, relax if too small
    for thr in (84, 56, 28, 0):
        ref_idx = np.where(first.values <= d-thr)[0]
        if len(ref_idx) >= 300: break
    q_idx = idx.get_indexer(hh)
    miss = q_idx < 0
    if miss.any(): q_idx[miss] = 0
    Xr, yr = X[ref_idx], sp_c[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)
    out = {}
    for k in (25, 50):
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        out[f'knn_med{k}'] = np.median(yr[nn], axis=1)
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    df = pd.DataFrame(out, index=hh)
    if miss.any(): df.loc[hh[miss]] = np.nan
    return df

for d in (95, 431):
    t0=time.time()
    v = api.snapshot(as_of_day=d)
    pf = peer_feats(v, d)
    tt = api.train_targets()
    e15 = api.load_saved('e015_stack.parquet')
    sd_ = e15[e15.snapshot_day==d].set_index('household_key')[['stack_ridge']]
    mm = pf.join(sd_, how='inner').join(tt[tt.snapshot_day==d].set_index('household_key').future_spend_4w, how='inner')
    yy = mm.future_spend_4w
    print(f'd={d} n={len(mm)} refs_ok time={time.time()-t0:.1f}s  stack MAE: {np.abs(mm.stack_ridge-yy).mean():.2f}')
    for c in pf.columns:
        print('  ', c, round(np.abs(mm[c]-yy).mean(),2))
    for w in (0.4,0.5,0.6):
        print(f'   blend w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med25-yy).mean(),2))
