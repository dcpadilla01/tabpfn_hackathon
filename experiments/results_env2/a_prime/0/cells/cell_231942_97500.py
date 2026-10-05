import agent_api, pandas as pd, numpy as np
print("snapshot_days:", agent_api.snapshot_days())
v = agent_api.snapshot(95)
print("view day/week:", v.day, v.week)
hh = v.households
print("households type:", type(hh), "len:", len(hh))
try:
    print("sample:", list(hh)[:5])
except Exception as e:
    print("iter err:", e)
tx = v.table("transactions")
print("tx shape:", tx.shape, "day max:", tx.day.max())
print(tx.head(3))
tt = agent_api.train_targets()
print("targets:", tt.shape)
print(tt.head(3))
d = v.table("demographics")
print("demo shape:", d.shape)
b = agent_api.load_saved("e012_outcome2.parquet")
print("e012 shape:", b.shape)
cols = list(b.columns)
print("n cols:", len(cols))
print("first 35:", cols[:35])
print("te2/outcome cols:", [c for c in cols if 'te2' in c or 'o2' in c or 'outcome' in c])
