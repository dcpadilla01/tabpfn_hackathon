l = load_saved('lvl_v1.parquet')
print(l.shape, l.columns.tolist())
sub = l[l.snapshot_day == 95]
hh0 = sub.household_key.iloc[0]
row = sub[sub.household_key == hh0].iloc[0]
v95 = snapshot(95)
tr = v95.transactions
trh = tr[(tr.household_key == hh0) & (tr.day > 95 - 364) & (tr.day <= 95)]
print('hh', hh0)
print('recomputed nprod_364', trh.product_id.nunique(), '| saved', row.nprod_364)
print('recomputed nbask_364', trh.basket_id.nunique(), '| saved', row.nbask_364)
st = trh.groupby('store_id').sales_value.sum()
print('recomputed top-store share %.4f' % (st.max() / st.sum()), '| saved store_share %.4f' % row.store_share)
print('recomputed sales/qty %.4f' % (trh.sales_value.sum() / trh.quantity.sum()), '| saved unit_price_364 %.4f' % row.unit_price_364)
# also check a validation-snapshot row for plausibility (no leak check possible directly, but verify shape/coverage)
print(l.groupby('snapshot_day').size())
print('NaN frac:', l.isna().mean().round(3).to_dict())
