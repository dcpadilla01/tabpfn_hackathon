t = load_saved('e011_table.parquet')
cols = sorted(c for c in t.columns if c not in ('household_key','snapshot_day'))
print('\n'.join(cols))
