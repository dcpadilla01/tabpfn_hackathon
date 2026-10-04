import numpy as np, pandas as pd
DEPTS = ['GROCERY','DRUG GM','MEAT','PRODUCE','KIOSK-GAS','MEAT-PCKGD','DELI','PASTRY','MISC SALES TRAN','NUTRITION']
def fn(view, snapshot_day):
    hh = pd.Index(view.households, name='household_key')
    tx = view.transactions.merge(view.products[['product_id','department','brand']], on='product_id', how='left')
    out = pd.DataFrame(index=hh)
    w84 = tx[tx.day > snapshot_day - 84]
    g = w84.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    tot = g.sum(axis=1).replace(0, np.nan)
    for d in DEPTS:
        out[f'share84_{d}'] = (g[d]/tot).fillna(0.0) if d in g.columns else 0.0
    out['div84'] = (g > 0).sum(axis=1).astype(float)
    out['max_share84'] = g.div(tot).max(axis=1).fillna(0.0)
    b = w84.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    btot = b.sum(axis=1).replace(0, np.nan)
    out['private_share84'] = (b['Private']/btot).fillna(0.0) if 'Private' in b.columns else 0.0
    w28 = tx[tx.day > snapshot_day - 28]
    g28 = w28.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0).reindex(hh).fillna(0.0)
    for d in DEPTS:
        out[f'spend28_{d}'] = g28[d] if d in g28.columns else 0.0
    return out
df1 = load_saved('e001_txhist.parquet')
df1['evening_share84'] = df1['evening_share84'].fillna(0.0)
bf = build_features(fn)
m = df1.merge(bf, on=['household_key','snapshot_day'], how='inner')
print("merged:", m.shape, "nans:", int(m.isna().sum().sum()))
assert m.shape[0] == 36426 and not m.isna().any().any()
path = save_table(m, 'e003_catmix')
print("saved:", path)
