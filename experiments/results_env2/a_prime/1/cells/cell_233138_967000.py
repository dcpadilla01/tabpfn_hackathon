import agent_api, pandas as pd, numpy as np, re

snap = agent_api.snapshot(459)
tx, prod = snap.transactions, snap.products
dep = prod.set_index('product_id')['department']
ds = tx['sales_value'].groupby(tx['product_id'].map(dep)).sum().sort_values(ascending=False)
print(ds.head(16)); print('n_depts', dep.nunique())
TOP = list(ds.index[:12]); TOPSET = set(TOP)
DEPG = dep.where(dep.isin(TOPSET), 'OTHER')  # product_id -> dept group
print('TOP:', TOP)

def clean(s): return re.sub(r'[^A-Za-z0-9]+', '_', str(s))[:20]
TOPC = [clean(c) for c in TOP] + ['OTHER']
COLS = TOP + ['OTHER']

def fn(view, snapshot_day):
    tx = view.table('transactions')
    d = tx[['household_key','day','product_id','sales_value']].copy()
    d['deptg'] = d['product_id'].map(DEPG).fillna('OTHER')
    hh = view.households
    out = pd.DataFrame(index=hh)
    def agg(w0, w1, tag, shares=False):
        sub = d[(d['day'] > w0) & (d['day'] <= w1)]
        if len(sub):
            piv = sub.groupby(['household_key','deptg'])['sales_value'].sum().unstack(fill_value=0.0)
            piv = piv.reindex(hh).reindex(columns=COLS, fill_value=0.0).fillna(0.0)
        else:
            piv = pd.DataFrame(0.0, index=hh, columns=COLS)
        for c, cc in zip(COLS, TOPC):
            out[f'{tag}_{cc}'] = piv[c]
        if shares:
            tot = piv.sum(axis=1).replace(0, np.nan)
            for c, cc in zip(TOP, TOPC[:-1]):
                out[f'{tag}sh_{cc}'] = (piv[c]/tot).fillna(0.0)
    agg(snapshot_day-28, snapshot_day, 'dsp28', shares=True)
    agg(snapshot_day-84, snapshot_day, 'dsp84')
    agg(snapshot_day-363, snapshot_day-336, 'dspL364')  # same 4w block one year earlier
    return out

feat = agent_api.build_features(fn)
print('feat', feat.shape)

old = agent_api.load_saved('e012_style.parquet')
print('old', old.shape)
overlap = (set(old.columns) & set(feat.columns)) - {'household_key','snapshot_day'}
print('overlap:', overlap)
m = old.merge(feat, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
path = agent_api.save_table(m, 'e013_dept.parquet')
print(path)
