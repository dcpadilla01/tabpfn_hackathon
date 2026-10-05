
df8 = load_saved('e008_level_shape.parquet')
print('E008 table:', df8.shape)
print(sorted([c for c in df8.columns if c not in ('household_key','snapshot_day')]))

e1 = load_saved('e001_history.parquet'); e3 = load_saved('e003_full.parquet')
e6 = load_saved('e006_cadence.parquet'); e7 = load_saved('e007_temporal.parquet')
df2 = load_saved('e002_mix.parquet')
print('\nE002 mix-only cols:', sorted(set(df2.columns)-set(e1.columns)-{'household_key','snapshot_day'}))
print('\nE006 cadence-only cols:', sorted(set(e6.columns)-set(e3.columns)-{'household_key','snapshot_day'}))
print('\nE007 temporal-only cols:', sorted(set(e7.columns)-set(e3.columns)-{'household_key','snapshot_day'}))

v = snapshot()
t = v.transactions
print('\ndiscount signs:')
print(t[['sales_value','quantity','retail_disc','coupon_disc','coupon_match_disc']].describe().loc[['mean','min','max']])
