s = agent_api.snapshot()
t = s.transactions
print("txn", t.shape, "hh", t.household_key.nunique(), "days", t.day.min(), t.day.max())
print(t.head(3))
p = s.products
print("products", p.shape, "depts", p.department.nunique())
print(p.department.value_counts().head(30))
print(p.brand.value_counts(dropna=False))
tt = agent_api.train_targets()
print(tt.future_spend_4w.describe())
print("rows per snapshot:")
print(tt.snapshot_day.value_counts().sort_index())
ct = s.table('campaign_targets')
print(ct.description.value_counts())
print(s.table('coupon_redemptions').shape)


# ---- cell ----
import numpy as np, pandas as pd

def mix_fn(view, snapshot_day):
    hh = view.households
    t = view.table('transactions')
    p = view.table('products')[['product_id','department','brand']]
    t = t.merge(p, on='product_id', how='left')
    t['dept'] = t.department.fillna('UNK')
    t['brand'] = t.brand.fillna('UNK')
    g = t.groupby(['household_key','dept']).sales_value.sum().unstack(fill_value=0.0)
    tot = t.groupby('household_key').sales_value.sum()
    mix = g.div(tot, axis=0)
    mix.columns = ['mix_' + c.lower().replace('.','_').replace('/','_').replace(' ','_') for c in mix.columns]
    # brand share
    b = t.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0)
    b = b.div(tot, axis=0)
    b.columns = ['share_brand_' + c.lower() for c in b.columns]
    # distinct depts visited in last 28d
    t28 = t[t.day > snapshot_day - 28]
    nd = t28.groupby('household_key').dept.nunique().rename('n_depts_28d')
    # store concentration last 56d
    t56 = t[t.day > snapshot_day - 56]
    st = t56.groupby(['household_key','store_id']).sales_value.sum()
    hhi = (st**2).groupby('household_key').sum() / (t56.groupby('household_key').sales_value.sum()**2)
    hhi = hhi.rename('store_hhi_56d')
    out = pd.concat([mix, b, nd, hhi], axis=1)
    out = out.reindex(hh)
    out['n_depts_28d'] = out.n_depts_28d.fillna(0)
    out['store_hhi_56d'] = out.store_hhi_56d.fillna(1.0)
    return out

feats = agent_api.build_features(mix_fn)
base = agent_api.load_saved('e001_rfm.parquet')
demo = agent_api.snapshot().table('demographics')
merged = base.merge(feats, on=['household_key','snapshot_day'], how='left').merge(demo, on='household_key', how='left')
print(merged.shape)
print([c for c in merged.columns if c.startswith('mix_')][:50])
print(merged[[c for c in merged.columns if c.startswith('mix_')]].isna().mean().mean())
path = agent_api.save_table(merged, 'e002_mix')
print(path)


# ---- cell ----
import numpy as np, pandas as pd

def mix_fn(view, snapshot_day):
    hh = view.households
    t = view.table('transactions')
    p = view.table('products')[['product_id','department','brand']]
    t = t.merge(p, on='product_id', how='left')
    t['dept'] = t.department.astype('object').where(t.department.notna(), 'UNK')
    t['brand'] = t.brand.astype('object').where(t.brand.notna(), 'UNK')
    g = t.groupby(['household_key','dept']).sales_value.sum().unstack(fill_value=0.0)
    tot = t.groupby('household_key').sales_value.sum()
    mix = g.div(tot, axis=0)
    mix.columns = ['mix_' + c.lower().replace('.','_').replace('/','_').replace(' ','_') for c in mix.columns]
    b = t.groupby(['household_key','brand']).sales_value.sum().unstack(fill_value=0.0)
    b = b.div(tot, axis=0)
    b.columns = ['share_brand_' + c.lower() for c in b.columns]
    t28 = t[t.day > snapshot_day - 28]
    nd = t28.groupby('household_key').dept.nunique().rename('n_depts_28d')
    t56 = t[t.day > snapshot_day - 56]
    st = t56.groupby(['household_key','store_id']).sales_value.sum()
    hhi = (st**2).groupby('household_key').sum() / (t56.groupby('household_key').sales_value.sum()**2)
    hhi = hhi.rename('store_hhi_56d')
    out = pd.concat([mix, b, nd, hhi], axis=1)
    out = out.reindex(hh)
    out['n_depts_28d'] = out.n_depts_28d.fillna(0)
    out['store_hhi_56d'] = out.store_hhi_56d.fillna(1.0)
    return out

feats = agent_api.build_features(mix_fn)
base = agent_api.load_saved('e001_rfm.parquet')
demo = agent_api.snapshot().table('demographics')
merged = base.merge(feats, on=['household_key','snapshot_day'], how='left').merge(demo, on='household_key', how='left')
print(merged.shape)
print(merged[[c for c in merged.columns if c.startswith('mix_')]].isna().mean().mean())
path = agent_api.save_table(merged, 'e002_mix')
print(path)
