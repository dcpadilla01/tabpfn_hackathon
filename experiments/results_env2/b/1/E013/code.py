import numpy as np, pandas as pd
print(snapshot_days())
tt = train_targets()
print('train rows', tt.shape)
print(tt.future_spend_4w.describe())
print('zero_share', float((tt.future_spend_4w==0).mean()), 'median', float(tt.future_spend_4w.median()))

e8 = load_saved('e008_level_shape.parquet')
print('e008', e8.shape)
cols8 = list(e8.columns)
kw = [c for c in cols8 if any(k in c.lower() for k in ['since','recen','tenure','last','gap','zero','streak','dorm','active'])]
print('recency-ish cols in e008:', kw)

for n in ['e006_cadence.parquet','e007_temporal.parquet','candAll.parquet','candA_dm.parquet','candB_g.parquet','candC_hb.parquet','candD_su.parquet','candE_td.parquet','candGST.parquet']:
    try:
        df = load_saved(n)
        newc = [c for c in df.columns if c not in cols8 and c not in ('household_key','snapshot_day')]
        print('==', n, df.shape, 'newcols', len(newc))
        print(newc[:70])
    except Exception as e:
        print(n, 'ERR', repr(e))

v = snapshot()
tx = v.transactions; pr = v.products
m = tx.merge(pr[['product_id','department','brand']], on='product_id', how='left')
print('dept spend top12:'); print(m.groupby('department').sales_value.sum().sort_values(ascending=False).head(12))
print('brand values:'); print(pr.brand.value_counts(dropna=False).head())

base = e8.copy()
for n in ['e006_cadence.parquet','e007_temporal.parquet']:
    df = load_saved(n)
    newc = [c for c in df.columns if c not in base.columns and c not in ('household_key','snapshot_day')]
    base = base.merge(df[['household_key','snapshot_day']+newc], on=['household_key','snapshot_day'], how='left')
print('union shape', base.shape, 'rows==e8:', len(base)==len(e8))
p = save_table(base, 'e013_union.parquet')
print('saved', p)
