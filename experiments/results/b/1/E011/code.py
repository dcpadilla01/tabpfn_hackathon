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


# ---- cell ----
import agent_api as A
df = A.load_saved('e012_full.parquet')
print('hh_id dtype:', df.hh_id.dtype, 'nunique:', df.hh_id.nunique())
print(df.hh_id.head(3).tolist())
# overlap with e011 columns
e11 = set(A.load_saved('e011_display.parquet').columns)
e12 = set(df.columns)
print('in e012 not e011:', sorted(e12 - e11))
print('in e011 not e012:', sorted(e11 - e12))
