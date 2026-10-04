import agent_api, numpy as np, pandas as pd

def peer_stats(day, k=10):
    snap = agent_api.snapshot(day)
    tx = snap.transactions
    prod = snap.products[['product_id','department','commodity_desc']]
    t = tx.merge(prod, on='product_id', how='left')
    tt = agent_api.train_targets()
    y = tt[tt.snapshot_day==day].set_index('household_key')['future_spend_4w']
    hh = y.index
    t = t[t.household_key.isin(hh)]
    out = {}
    r28 = t[t.day>day-28]
    out['spend28'] = r28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0).corr(y)
    out['trips28']  = r28.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0).corr(y)
    out['recency']  = tx.groupby('household_key').day.max().rsub(day).reindex(hh).corr(y)
    # basket-size / timing baselines
    b = r28.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'), val=('sales_value','sum'))
    out['lines_per_basket'] = b.groupby('h').lines.mean().reindex(hh).fillna(0).corr(y)
    out['val_per_basket']   = b.groupby('h').val.mean().reindex(hh).fillna(0).corr(y)
    tt28 = r28[['household_key','trans_time']].dropna()
    out['evening_share'] = tt28.assign(ev=(tt28.trans_time>=1700).astype(float)).groupby('household_key').ev.mean().reindex(hh).fillna(0).corr(y)
    # peer kNN on spend-share vectors
    for col, win in [('department',112),('department',364),('commodity_desc',112)]:
        tw = t[t.day>day-win]
        piv = tw.pivot_table(index='household_key', columns=col, values='sales_value', aggfunc='sum').fillna(0.0).reindex(hh).fillna(0.0)
        M = piv.values.astype(float)
        Mn = M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9)
        sim = Mn@Mn.T
        np.fill_diagonal(sim,0.0)
        sp28v = r28.groupby('household_key').sales_value.sum().reindex(hh).fillna(0).values
        idx = np.argpartition(-sim, k-1, axis=1)[:,:k]
        w = np.take_along_axis(sim, idx, axis=1)
        w = w/np.maximum(w.sum(1,keepdims=True),1e-9)
        peer = pd.Series((w*sp28v[idx]).sum(1), index=hh)
        out[f'peer_{col[:4]}_w{win}_k{k}'] = peer.corr(y)
    print('n depts:', t.department.nunique(), 'n commodities:', t.commodity_desc.nunique(), 'n hh:', len(hh))
    return {a: round(v,3) for a,v in out.items()}

for d in (347, 431):
    print(d, peer_stats(d))
