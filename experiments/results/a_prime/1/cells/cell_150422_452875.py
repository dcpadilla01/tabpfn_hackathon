t3 = agent_api.load_saved('e003_catmix.parquet')
t4 = agent_api.load_saved('e004_mkt.parquet')
print('e003:', t3.shape, 'e004:', t4.shape)
key = ['household_key', 'snapshot_day']
dup = [c for c in t4.columns if c in t3.columns and c not in key]
print('dup cols dropped from e004:', dup)
m = t3.merge(t4.drop(columns=dup), on=key, how='inner')
print('merged:', m.shape)
p = agent_api.save_table(m, 'e006_catmix_mkt')
print('saved:', p)