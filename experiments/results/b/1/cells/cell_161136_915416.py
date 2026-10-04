import agent_api as api
names = ['e012_full','e009_macro','e008_decomp2','e004_long_hist','e006_seq_gaps','e011_display']
for nm in names:
    t = api.load_saved(nm + '.parquet')
    print('===', nm, t.shape)
    print([c for c in t.columns if c not in ('household_key','snapshot_day')])
    print()