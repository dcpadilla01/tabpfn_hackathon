df = load_saved('e011_price.parquet')
print('e011 shape', df.shape)
cols = [c for c in df.columns if c not in ('household_key','snapshot_day')]
print('n features', len(cols))
for c in cols: print(c)
print()
t = train_targets()
y = t[TARGET]
print(y.describe())
print('zero frac', round(float((y==0).mean()),4), 'frac>200', round(float((y>200).mean()),4))
r = load_saved('rfm28.parquet'); print('rfm cols', [c for c in r.columns if c not in ('household_key','snapshot_day')])
m = t.merge(r, on=['household_key','snapshot_day'], how='left')
s = m['spend28'].fillna(0.0)
print('corr spend28', round(float(np.corrcoef(s, y)[0,1]),3), 'corr log1p(spend28)', round(float(np.corrcoef(np.log1p(s), y)[0,1]),3))
d = load_saved('mkt_demo.parquet')
dc = [c for c in d.columns if c not in ('household_key','snapshot_day')]
print('mkt_demo n cols', len(dc)); print(dc)