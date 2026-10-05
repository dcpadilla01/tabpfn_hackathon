import pandas as pd, numpy as np
import agent_api

e011 = agent_api.load_saved('e011_table.parquet')
d = e011.copy()
snap = agent_api.snapshot(459)
tx = snap.transactions
prod = snap.products
sz = pd.to_numeric(prod['curr_size_of_product'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
pmap_sz = pd.Series(sz.values, index=prod['product_id'].values)
tx2 = tx[['household_key','day','sales_value','quantity','store_id','product_id','basket_id']].copy()
tx2['up'] = tx2['sales_value']/tx2['quantity'].replace(0,np.nan)
tx2['psz'] = tx2['product_id'].map(pmap_sz)
gg = tx2.groupby('household_key')
hh = d['household_key'].values; S = d['snapshot_day'].values
cols = ['store_share','unit_price_84','unit_price_364','psz_med_364','nprod_364','nbask_364','top_store']
out = {c: [] for c in cols}
for k, s in zip(hh, S):
    idx = gg.indices.get(k)
    if idx is None or len(idx)==0:
        for c in cols: out[c].append(np.nan)
        continue
    dd = tx2['day'].values[idx]; st = tx2['store_id'].values[idx]
    sv = tx2['sales_value'].values[idx]; up = tx2['up'].values[idx]; pz = tx2['psz'].values[idx]
    pid = tx2['product_id'].values[idx]; bk = tx2['basket_id'].values[idx]
    m364 = (dd > s-364) & (dd <= s); m84 = (dd > s-84) & (dd <= s)
    if m364.sum()==0:
        for c in cols: out[c].append(np.nan)
        continue
    cs = pd.Series(sv[m364]).groupby(st[m364]).sum()
    out['top_store'].append(float(cs.idxmax()))
    out['store_share'].append(float(cs.max()/cs.sum()))
    u84 = up[m84]; u84 = u84[np.isfinite(u84)]
    out['unit_price_84'].append(float(np.median(u84)) if len(u84) else np.nan)
    u364 = up[m364]; u364 = u364[np.isfinite(u364)]
    out['unit_price_364'].append(float(np.median(u364)) if len(u364) else np.nan)
    pzv = pz[m364]; pzv = pzv[np.isfinite(pzv)]
    out['psz_med_364'].append(float(np.median(pzv)) if len(pzv) else np.nan)
    out['nprod_364'].append(float(len(np.unique(pid[m364]))))
    out['nbask_364'].append(float(len(np.unique(bk[m364]))))
L = pd.DataFrame(out, index=d.index).astype(float)
ts_counts = L['top_store'].value_counts()
top_stores = list(ts_counts.index[:12])
for s_ in top_stores: L[f'store_{int(s_)}'] = (L['top_store']==s_).astype(float)
L['store_other'] = (~L['top_store'].isin(top_stores)).astype(float)
L = L.drop(columns=['top_store'])
L.insert(0,'household_key', d['household_key'].values)
L.insert(1,'snapshot_day', d['snapshot_day'].values)
print('lvl table', L.shape, list(L.columns)[:10])
path = agent_api.save_table(L, 'lvl_v1')
print('PATH:', path)
