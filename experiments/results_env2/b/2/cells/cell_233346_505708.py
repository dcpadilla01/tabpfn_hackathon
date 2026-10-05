m = load_saved('mkt_v1.parquet')
mc = [c for c in m.columns if c not in ('household_key','snapshot_day')]
base = set(load_saved('churn_vol_v1.parquet').columns)
print('mkt-only feats:', [c for c in mc if c not in base])
cv = load_saved('churn_vol_v1.parquet')
bc = [c for c in cv.columns if c.startswith('b_')]
print('b_* feats:', bc)
