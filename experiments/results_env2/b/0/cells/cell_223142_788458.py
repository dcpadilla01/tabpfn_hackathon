import agent_api as A
for name in ['e006_zero_inflation.parquet','e006_newblock.parquet']:
    df = A.load_saved(name)
    print(name, df.shape)
    print(sorted(df.columns)[:40])
    print('...')
    print(sorted(df.columns)[40:])
    print()
