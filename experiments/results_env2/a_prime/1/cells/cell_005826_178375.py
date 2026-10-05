import agent_api as api
import numpy as np, pandas as pd, time

def peer_feats(view, d, ks=(25,50,100)):
    tx = view.table('transactions')
    hh = pd.Index(view.households)
    g = tx.groupby('household_key')
    # current window [d-27, d], previous [d-55, d-28]
    cur = tx[(tx.day>=d-27)&(tx.day<=d)]
    prv = tx[(tx.day>=d-55)&(tx.day<=d-28)]
    sp_cur = cur.groupby('household_key').sales_value.sum()
    tr_cur = cur.groupby('household_key').basket_id.nunique()
    sp_prv = prv.groupby('household_key').sales_value.sum()
    tr_prv = prv.groupby('household_key').basket_id.nunique()
    first = g.day.min(); last = g.day.max()
    idx = first.index
    rec = (d - last).clip(lower=0).reindex(idx).fillna(999).values.astype(float)
    ten = (d - first + 1).values.astype(float)
    X = np.column_stack([
        np.log1p(sp_cur.reindex(idx).fillna(0).values),
        np.log1p(tr_cur.reindex(idx).fillna(0).values),
        np.log1p(rec), np.log1p(ten),
        np.log1p(sp_prv.reindex(idx).fillna(0).values)])
    realized = sp_cur.reindex(idx).fillna(0).values  # for refs: spend d-27..d
    ref_mask = (first.values <= d-112) & (first.index.isin(idx))
    ref_idx = np.where(first.values <= d-112)[0]
    q_idx = idx.get_indexer(hh)
    Xr = X[ref_idx]; yr = realized[ref_idx]
    mu, sd = Xr.mean(0), Xr.std(0)+1e-9
    Zr = (Xr-mu)/sd; Zq = (X[q_idx]-mu)/sd
    out = {}
    nq = Zq.shape[0]
    d2 = ((Zq[:,None,:]-Zr[None,:,:])**2).sum(-1)  # may be big: nq~2500 x nr~2500 fine
    for k in ks:
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        med = np.median(yr[nn], axis=1)
        out[f'knn_med{k}'] = med
        if k==50:
            out['knn_mean50'] = yr[nn].mean(1)
            out['knn_medlog50'] = np.expm1(np.median(np.log1p(yr[nn]), axis=1))
    # drift ratio: median realized/sp_prv over refs
    spp = sp_prv.values[ref_idx]
    ratio = yr[spp>0]/spp[spp>0]
    out['drift_r'] = np.full(nq, np.median(ratio))
    return pd.DataFrame(out, index=hh)

t0=time.time()
v = api.snapshot(as_of_day=431)
pf = peer_feats(v, 431)
print('time:', round(time.time()-t0,1),'s'); print(pf.describe().round(2))

# quick quality check vs train targets at d=431
tt = api.train_targets()
m = pf.join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
y = m.future_spend_4w
for c in pf.columns:
    print(c, 'MAE:', round(np.abs(m[c]-y).mean(),2))
b = 0.5*m.knn_med50+0.5*api.load_saved('e015_stack.parquet').set_index(['household_key','snapshot_day']).loc[(slice(None),431),:]['stack_ridge'].values if False else None
e15 = api.load_saved('e015_stack.parquet')
s431 = e15[e15.snapshot_day==431].set_index('household_key').stack_ridge
mm = pf.join(s431, how='inner').join(tt[tt.snapshot_day==431].set_index('household_key').future_spend_4w, how='inner')
yy = mm.future_spend_4w
print('stack MAE 431:', round(np.abs(mm.stack_ridge-yy).mean(),2))
for w in [0.3,0.5,0.7]:
    print(f'blend w={w}:', round(np.abs(w*mm.stack_ridge+(1-w)*mm.knn_med50-yy).mean(),2))
