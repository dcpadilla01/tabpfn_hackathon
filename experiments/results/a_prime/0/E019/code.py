import numpy as np, pandas as pd, agent_api

COLS = ['kn_s28_w10','kn_s28_m10','kn_s84_w10','kn_s84_m10','kn_b28_w10',
        'kn_act10','kn_act5','kn_s28_w5','kn_s28_m5','kn_ratio','kn_nbr_sim','kn_has_nbr']

def knn_block(view, snapshot_day):
    hh_need = pd.Index(view.households)
    tx = view.transactions
    out = pd.DataFrame(0.0, index=hh_need, columns=COLS)
    if len(tx) == 0:
        return out
    prod = view.products
    if prod is not None and len(prod) > 0:
        tx = tx.merge(prod[['product_id','department']], on='product_id', how='left')
        tx['department'] = tx['department'].fillna('UNK')
    else:
        tx['department'] = 'ALL'
    w84 = tx[(tx.day > snapshot_day-84) & (tx.day <= snapshot_day)]
    w28 = tx[(tx.day > snapshot_day-28) & (tx.day <= snapshot_day)]
    piv = w84.pivot_table(index='household_key', columns='department',
                          values='sales_value', aggfunc='sum').fillna(0.0)
    hh84 = piv.index
    M = piv.values.astype(float)
    n = M.shape[0]
    norms = np.linalg.norm(M, axis=1)
    nz = norms > 0
    Mn = np.zeros_like(M)
    Mn[nz] = M[nz] / norms[nz][:, None]
    S = Mn @ Mn.T
    np.fill_diagonal(S, 0.0)
    k = min(10, max(1, n-1))
    idx = np.argpartition(-S, k-1, axis=1)[:, :k]
    sim = np.take_along_axis(S, idx, axis=1)
    s28 = w28.groupby('household_key').sales_value.sum().reindex(hh84).fillna(0.0).values
    s84 = w84.groupby('household_key').sales_value.sum().reindex(hh84).fillna(0.0).values
    b28 = w28.groupby('household_key').basket_id.nunique().reindex(hh84).fillna(0.0).values
    ws = sim.sum(1); wsn = np.where(ws > 0, ws, 1.0)
    f = {}
    f['kn_s28_w10'] = (sim * s28[idx]).sum(1) / wsn
    f['kn_s28_m10'] = s28[idx].mean(1)
    f['kn_s84_w10'] = (sim * s84[idx]).sum(1) / wsn
    f['kn_s84_m10'] = s84[idx].mean(1)
    f['kn_b28_w10'] = (sim * b28[idx]).sum(1) / wsn
    f['kn_act10'] = (s28[idx] > 0).mean(1)
    idx5, sim5 = idx[:, :5], sim[:, :5]
    ws5 = sim5.sum(1); ws5n = np.where(ws5 > 0, ws5, 1.0)
    f['kn_act5'] = (s28[idx5] > 0).mean(1)
    f['kn_s28_w5'] = (sim5 * s28[idx5]).sum(1) / ws5n
    f['kn_s28_m5'] = s28[idx5].mean(1)
    f['kn_ratio'] = s28 / (f['kn_s28_w10'] + 5.0)
    f['kn_nbr_sim'] = np.where(ws > 0, ws / k, 0.0)
    F = pd.DataFrame(f, index=hh84)
    F['kn_has_nbr'] = (ws > 0).astype(float)
    F = F.reindex(hh_need)
    F['kn_has_nbr'] = F['kn_has_nbr'].fillna(0.0)
    return F.fillna(0.0)

base = agent_api.load_saved('e013_stock.parquet')
blk = agent_api.build_features(knn_block)
print('base', base.shape, 'blk', blk.shape)
m = agent_api.KEYS
m = base.merge(blk, on=['household_key', 'snapshot_day'], how='left')
newcols = [c for c in blk.columns if c not in ('household_key', 'snapshot_day')]
print('merged', m.shape, 'nan in new:', int(m[newcols].isna().sum().sum()))
path = agent_api.save_table(m, 'e019_knn.parquet')
print(path)
print(m[newcols].describe().loc[['mean','std']].T)

# ---- cell ----
import numpy as np, pandas as pd, agent_api

COLS = ['kn_s28_w10','kn_s28_m10','kn_s84_w10','kn_s84_m10','kn_b28_w10',
        'kn_act10','kn_act5','kn_s28_w5','kn_s28_m5','kn_ratio','kn_nbr_sim','kn_has_nbr']

def knn_block(view, snapshot_day):
    hh_need = pd.Index(view.households)
    tx = view.transactions
    out = pd.DataFrame(0.0, index=hh_need, columns=COLS)
    if len(tx) == 0:
        return out
    prod = view.products
    if prod is not None and len(prod) > 0:
        tx = tx.merge(prod[['product_id','department']], on='product_id', how='left')
        tx['department'] = tx['department'].astype(str).fillna('UNK')
    else:
        tx['department'] = 'ALL'
    w84 = tx[(tx.day > snapshot_day-84) & (tx.day <= snapshot_day)]
    w28 = tx[(tx.day > snapshot_day-28) & (tx.day <= snapshot_day)]
    piv = w84.pivot_table(index='household_key', columns='department',
                          values='sales_value', aggfunc='sum').fillna(0.0)
    hh84 = piv.index
    M = piv.values.astype(float)
    norms = np.linalg.norm(M, axis=1)
    nz = norms > 0
    Mn = np.zeros_like(M)
    Mn[nz] = M[nz] / norms[nz][:, None]
    S = Mn @ Mn.T
    np.fill_diagonal(S, 0.0)
    k = min(10, max(1, n-1)) if (n := M.shape[0]) > 1 else 1
    k = min(k, n-1) if n > 1 else 1
    idx = np.argpartition(-S, k-1, axis=1)[:, :k]
    sim = np.take_along_axis(S, idx, axis=1)
    s28 = w28.groupby('household_key').sales_value.sum().reindex(hh84).fillna(0.0).values
    s84 = w84.groupby('household_key').sales_value.sum().reindex(hh84).fillna(0.0).values
    b28 = w28.groupby('household_key').basket_id.nunique().reindex(hh84).fillna(0.0).values
    ws = sim.sum(1); wsn = np.where(ws > 0, ws, 1.0)
    f = {}
    f['kn_s28_w10'] = (sim * s28[idx]).sum(1) / wsn
    f['kn_s28_m10'] = s28[idx].mean(1)
    f['kn_s84_w10'] = (sim * s84[idx]).sum(1) / wsn
    f['kn_s84_m10'] = s84[idx].mean(1)
    f['kn_b28_w10'] = (sim * b28[idx]).sum(1) / wsn
    f['kn_act10'] = (s28[idx] > 0).mean(1)
    idx5, sim5 = idx[:, :5], sim[:, :5]
    ws5 = sim5.sum(1); ws5n = np.where(ws5 > 0, ws5, 1.0)
    f['kn_act5'] = (s28[idx5] > 0).mean(1)
    f['kn_s28_w5'] = (sim5 * s28[idx5]).sum(1) / ws5n
    f['kn_s28_m5'] = s28[idx5].mean(1)
    f['kn_ratio'] = s28 / (f['kn_s28_w10'] + 5.0)
    f['kn_nbr_sim'] = np.where(ws > 0, ws / k, 0.0)
    F = pd.DataFrame(f, index=hh84)
    F['kn_has_nbr'] = (ws > 0).astype(float)
    F = F.reindex(hh_need)
    F['kn_has_nbr'] = F['kn_has_nbr'].fillna(0.0)
    return F.fillna(0.0)

base = agent_api.load_saved('e013_stock.parquet')
blk = agent_api.build_features(knn_block)
print('base', base.shape, 'blk', blk.shape)
m = base.merge(blk, on=['household_key', 'snapshot_day'], how='left')
newcols = [c for c in blk.columns if c not in ('household_key', 'snapshot_day')]
print('merged', m.shape, 'nan in new:', int(m[newcols].isna().sum().sum()))
path = agent_api.save_table(m, 'e019_knn.parquet')
print(path)
print(m[newcols].describe().loc[['mean','std']].T)