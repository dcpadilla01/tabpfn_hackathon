import agent_api as A

for nm in ['e011_display', 'e012_full', 'e009_macro']:
    for cand in [nm + '.parquet', nm]:
        try:
            df = A.load_saved(cand)
            break
        except Exception as e:
            df = None
            err = e
    if df is None:
        print(nm, 'LOAD ERR', err)
        continue
    print('===', nm, df.shape)
    print(list(df.columns))
    print('dup keys:', df.duplicated(['household_key', 'snapshot_day']).sum())
    print('snapdays:', sorted(df.snapshot_day.unique()))
    print(df.head(2).T)
    print()
