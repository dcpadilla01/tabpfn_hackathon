import agent_api as A
df = A.load_saved('e012_full.parquet')
print('hh_id dtype:', df.hh_id.dtype, 'nunique:', df.hh_id.nunique())
print(df.hh_id.head(3).tolist())
# overlap with e011 columns
e11 = set(A.load_saved('e011_display.parquet').columns)
e12 = set(df.columns)
print('in e012 not e011:', sorted(e12 - e11))
print('in e011 not e012:', sorted(e11 - e12))
