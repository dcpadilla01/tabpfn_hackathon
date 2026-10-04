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


# ---- cell ----
import agent_api
t = agent_api.load_saved('e011_price.parquet')
print(t.shape, len(t.columns)-2)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(', '.join(cols))


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_feats(view, day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions

    # peer stats over all households with history (all info <= day: no leakage)
    spend28 = tx[tx.day > day-28].groupby('household_key').sales_value.sum()
    trips28 = tx[tx.day > day-28].groupby('household_key').basket_id.nunique()
    g = tx.groupby('household_key')
    peer_stats = pd.DataFrame({
        'spend28': spend28, 'trips28': trips28,
        'recency': day - g.day.max(), 'tenure': day - g.day.min()})

    def knn_aggs(win, col, ks=(10,)):
        tw = tx[tx.day > day-win][['household_key', col, 'sales_value']].dropna()
        piv = tw.pivot_table(index='household_key', columns=col,
                             values='sales_value', aggfunc='sum').fillna(0.0)
        M = piv.values.astype(float)
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        S = Mn @ Mn.T
        np.fill_diagonal(S, -1.0)
        pidx = piv.index
        out = {}
        for k in ks:
            kk = min(k, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            ws = w.sum(1)
            for stat in peer_stats.columns:
                v = peer_stats[stat].reindex(pidx).values.astype(float)
                agg = np.where(ws > 1e-12, (w * v[idxp]).sum(1) / np.maximum(ws, 1e-12), np.nan)
                out[f'peer{k}_{stat}_' + ('d' if col == 'department' else 'c') + f'{win}'] = pd.Series(agg, index=pidx)
        if 10 in ks:
            kk = min(10, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            out['sim_max10'] = pd.Series(w.max(1), index=pidx)
            out['sim_mean10'] = pd.Series(np.where(w.sum(1) > 0, w.sum(1) / kk, np.nan), index=pidx)
            out['nn_pos'] = pd.Series((w > 0.15).sum(1).astype(float), index=pidx)
        return pd.DataFrame(out)

    P = pd.concat([knn_aggs(112, 'department', ks=(5, 10, 20)),
                   knn_aggs(364, 'department', ks=(10,)),
                   knn_aggs(112, 'commodity_desc', ks=(10,))], axis=1)
    P = P[~P.index.duplicated(keep='last')].reindex(hh)

    # basket-size, timing, dow, store, brand traits (own household)
    r28 = tx[tx.day > day-28]
    b = r28.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F = pd.DataFrame(index=hh)
    F['lines_basket28'] = b.groupby('h').lines.mean().reindex(hh)
    r112 = tx[tx.day > day-112]
    b112 = r112.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F['lines_basket112'] = b112.groupby('h').lines.mean().reindex(hh)
    tt = r28[['household_key','trans_time']].dropna().assign(t=lambda d: d.trans_time.astype(float))
    F['evening_share28'] = tt.assign(ev=(tt.t >= 1700).astype(float)).groupby('household_key').ev.mean().reindex(hh)
    F['morning_share28'] = tt.assign(mo=(tt.t < 1200).astype(float)).groupby('household_key').mo.mean().reindex(hh)
    F['tt_std28'] = tt.groupby('household_key').t.std().reindex(hh)
    F['ndow28'] = r28.assign(dw=r28.day % 7).groupby('household_key').dw.nunique().reindex(hh)
    st = r112.groupby(['household_key','store_id']).sales_value.sum()
    stt = st.groupby('household_key').agg(['sum','max'])
    F['store_top_share112'] = stt['max'] / stt['sum'].replace(0, np.nan)
    pr = st / st.groupby('household_key').transform('sum').replace(0, np.nan)
    F['store_entropy112'] = (-(pr * np.log(pr.clip(lower=1e-12))).groupby('household_key').sum()).reindex(hh)
    prod = view.products[['product_id','brand']]
    br = r112.merge(prod, on='product_id', how='left')
    bs = br.groupby('household_key').sales_value.sum()
    pv = br[br.brand == 'Private'].groupby('household_key').sales_value.sum()
    F['private_share112'] = (pv / bs.replace(0, np.nan)).reindex(hh)

    X = pd.concat([P, F], axis=1).reset_index()
    X['snapshot_day'] = day
    return X

out = agent_api.build_features(make_feats)
base = agent_api.load_saved('e011_price.parquet')
merged = base.merge(out, on=['household_key','snapshot_day'], how='inner')
print('merged shape:', merged.shape)
assert merged.shape[0] == base.shape[0] == out.shape[0]
path = agent_api.save_table(merged, 'e016_peer.parquet')
print(path)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_feats(view, day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    prod = view.products[['product_id','department','commodity_desc']]
    txp = tx.merge(prod, on='product_id', how='left')

    spend28 = tx[tx.day > day-28].groupby('household_key').sales_value.sum()
    trips28 = tx[tx.day > day-28].groupby('household_key').basket_id.nunique()
    g = tx.groupby('household_key')
    peer_stats = pd.DataFrame({
        'spend28': spend28, 'trips28': trips28,
        'recency': day - g.day.max(), 'tenure': day - g.day.min()})

    def knn_aggs(win, col, ks=(10,)):
        tw = txp[txp.day > day-win][['household_key', col, 'sales_value']].dropna()
        piv = tw.pivot_table(index='household_key', columns=col,
                             values='sales_value', aggfunc='sum').fillna(0.0)
        M = piv.values.astype(float)
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        S = Mn @ Mn.T
        np.fill_diagonal(S, -1.0)
        pidx = piv.index
        out = {}
        for k in ks:
            kk = min(k, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            ws = w.sum(1)
            for stat in peer_stats.columns:
                v = peer_stats[stat].reindex(pidx).values.astype(float)
                agg = np.where(ws > 1e-12, (w * v[idxp]).sum(1) / np.maximum(ws, 1e-12), np.nan)
                out[f'peer{k}_{stat}_' + ('d' if col == 'department' else 'c') + f'{win}'] = pd.Series(agg, index=pidx)
        if 10 in ks:
            kk = min(10, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            out['sim_max10'] = pd.Series(w.max(1), index=pidx)
            out['sim_mean10'] = pd.Series(np.where(w.sum(1) > 0, w.sum(1) / kk, np.nan), index=pidx)
            out['nn_pos'] = pd.Series((w > 0.15).sum(1).astype(float), index=pidx)
        return pd.DataFrame(out)

    P = pd.concat([knn_aggs(112, 'department', ks=(5, 10, 20)),
                   knn_aggs(364, 'department', ks=(10,)),
                   knn_aggs(112, 'commodity_desc', ks=(10,))], axis=1)
    P = P[~P.index.duplicated(keep='last')].reindex(hh)

    r28 = tx[tx.day > day-28]
    b = r28.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F = pd.DataFrame(index=hh)
    F['lines_basket28'] = b.groupby('h').lines.mean().reindex(hh)
    r112 = tx[tx.day > day-112]
    b112 = r112.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F['lines_basket112'] = b112.groupby('h').lines.mean().reindex(hh)
    tt = r28[['household_key','trans_time']].dropna().assign(t=lambda d: d.trans_time.astype(float))
    F['evening_share28'] = tt.assign(ev=(tt.t >= 1700).astype(float)).groupby('household_key').ev.mean().reindex(hh)
    F['morning_share28'] = tt.assign(mo=(tt.t < 1200).astype(float)).groupby('household_key').mo.mean().reindex(hh)
    F['tt_std28'] = tt.groupby('household_key').t.std().reindex(hh)
    F['ndow28'] = r28.assign(dw=r28.day % 7).groupby('household_key').dw.nunique().reindex(hh)
    st = r112.groupby(['household_key','store_id']).sales_value.sum()
    stt = st.groupby('household_key').agg(['sum','max'])
    F['store_top_share112'] = stt['max'] / stt['sum'].replace(0, np.nan)
    pr = st / st.groupby('household_key').transform('sum').replace(0, np.nan)
    F['store_entropy112'] = (-(pr * np.log(pr.clip(lower=1e-12))).groupby('household_key').sum()).reindex(hh)
    prod2 = view.products[['product_id','brand']]
    br = r112.merge(prod2, on='product_id', how='left')
    bs = br.groupby('household_key').sales_value.sum()
    pv = br[br.brand == 'Private'].groupby('household_key').sales_value.sum()
    F['private_share112'] = (pv / bs.replace(0, np.nan)).reindex(hh)

    X = pd.concat([P, F], axis=1).reset_index()
    X['snapshot_day'] = day
    return X

out = agent_api.build_features(make_feats)
base = agent_api.load_saved('e011_price.parquet')
merged = base.merge(out, on=['household_key','snapshot_day'], how='inner')
print('merged shape:', merged.shape)
assert merged.shape[0] == base.shape[0] == out.shape[0]
path = agent_api.save_table(merged, 'e016_peer.parquet')
print(path)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_feats(view, day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions
    prod = view.products[['product_id','department','commodity_desc']]
    txp = tx.merge(prod, on='product_id', how='left')

    spend28 = tx[tx.day > day-28].groupby('household_key').sales_value.sum()
    trips28 = tx[tx.day > day-28].groupby('household_key').basket_id.nunique()
    g = tx.groupby('household_key')
    peer_stats = pd.DataFrame({
        'spend28': spend28, 'trips28': trips28,
        'recency': day - g.day.max(), 'tenure': day - g.day.min()})

    def knn_aggs(win, col, ks=(10,)):
        tag = ('d' if col == 'department' else 'c') + str(win)
        tw = txp[txp.day > day-win][['household_key', col, 'sales_value']].dropna()
        piv = tw.pivot_table(index='household_key', columns=col,
                             values='sales_value', aggfunc='sum').fillna(0.0)
        M = piv.values.astype(float)
        Mn = M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
        S = Mn @ Mn.T
        np.fill_diagonal(S, -1.0)
        pidx = piv.index
        out = {}
        for k in ks:
            kk = min(k, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            ws = w.sum(1)
            for stat in peer_stats.columns:
                v = peer_stats[stat].reindex(pidx).values.astype(float)
                agg = np.where(ws > 1e-12, (w * v[idxp]).sum(1) / np.maximum(ws, 1e-12), np.nan)
                out[f'peer{k}_{stat}_{tag}'] = pd.Series(agg, index=pidx)
        if 10 in ks:
            kk = min(10, S.shape[1] - 1)
            idxp = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            w = np.clip(np.take_along_axis(S, idxp, axis=1), 0, None)
            out[f'sim_max_{tag}'] = pd.Series(w.max(1), index=pidx)
            out[f'sim_mean_{tag}'] = pd.Series(np.where(w.sum(1) > 0, w.sum(1) / kk, np.nan), index=pidx)
            out[f'nn_pos_{tag}'] = pd.Series((w > 0.15).sum(1).astype(float), index=pidx)
        return pd.DataFrame(out)

    P = pd.concat([knn_aggs(112, 'department', ks=(5, 10, 20)),
                   knn_aggs(364, 'department', ks=(10,)),
                   knn_aggs(112, 'commodity_desc', ks=(10,))], axis=1)
    P = P[~P.index.duplicated(keep='last')].reindex(hh)

    r28 = tx[tx.day > day-28]
    b = r28.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F = pd.DataFrame(index=hh)
    F['lines_basket28'] = b.groupby('h').lines.mean().reindex(hh)
    r112 = tx[tx.day > day-112]
    b112 = r112.groupby('basket_id').agg(h=('household_key','first'), lines=('product_id','count'))
    F['lines_basket112'] = b112.groupby('h').lines.mean().reindex(hh)
    tt = r28[['household_key','trans_time']].dropna().assign(t=lambda d: d.trans_time.astype(float))
    F['evening_share28'] = tt.assign(ev=(tt.t >= 1700).astype(float)).groupby('household_key').ev.mean().reindex(hh)
    F['morning_share28'] = tt.assign(mo=(tt.t < 1200).astype(float)).groupby('household_key').mo.mean().reindex(hh)
    F['tt_std28'] = tt.groupby('household_key').t.std().reindex(hh)
    F['ndow28'] = r28.assign(dw=r28.day % 7).groupby('household_key').dw.nunique().reindex(hh)
    st = r112.groupby(['household_key','store_id']).sales_value.sum()
    stt = st.groupby('household_key').agg(['sum','max'])
    F['store_top_share112'] = stt['max'] / stt['sum'].replace(0, np.nan)
    pr = st / st.groupby('household_key').transform('sum').replace(0, np.nan)
    F['store_entropy112'] = (-(pr * np.log(pr.clip(lower=1e-12))).groupby('household_key').sum()).reindex(hh)
    prod2 = view.products[['product_id','brand']]
    br = r112.merge(prod2, on='product_id', how='left')
    bs = br.groupby('household_key').sales_value.sum()
    pv = br[br.brand == 'Private'].groupby('household_key').sales_value.sum()
    F['private_share112'] = (pv / bs.replace(0, np.nan)).reindex(hh)

    X = pd.concat([P, F], axis=1).reset_index()
    X['snapshot_day'] = day
    return X

out = agent_api.build_features(make_feats)
base = agent_api.load_saved('e011_price.parquet')
merged = base.merge(out, on=['household_key','snapshot_day'], how='inner')
print('merged shape:', merged.shape, 'dup cols:', merged.columns.duplicated().sum())
assert merged.shape[0] == base.shape[0] == out.shape[0]
path = agent_api.save_table(merged, 'e016_peer.parquet')
print(path)
