
import agent_api, pandas as pd

# Inspect saved E011 table
t = agent_api.load_saved('e011_table.parquet')
print("e011_table shape:", t.shape)
cols = list(t.columns)
print("n cols:", len(cols))
print(cols)
print(t.head(3))

# Test whether load_saved works INSIDE build_features
def fn(view, snapshot_day):
    tt = agent_api.load_saved('e011_table.parquet')
    sub = tt[tt['snapshot_day'] == snapshot_day]
    return sub.set_index('household_key').drop(columns=['snapshot_day'])

bf = agent_api.build_features(fn)
print("rebuilt via build_features:", bf.shape)
print(bf.head(3))
