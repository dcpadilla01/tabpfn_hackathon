
base = load_saved('e011_discounts.parquet')
print(base.shape)
print(base.columns.tolist())
print(base.dtypes.value_counts())

# verify load_saved works inside build_features and rows line up
def fn(view, sd):
    b = load_saved('e011_discounts.parquet')
    b = b[b['snapshot_day'] == sd].set_index('household_key')
    return b.drop(columns=['snapshot_day'])

out = build_features(fn)
print('built:', out.shape)
print('n snapshots:', out.snapshot_day.nunique() if 'snapshot_day' in out.columns else 'n/a')
print(out.head(3))
