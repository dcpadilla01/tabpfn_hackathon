import pandas as pd, numpy as np
from agent_api import load_saved, train_targets, snapshot

for n in ["e003_dept_mix","e004_long_hist","e011_display","e002_marketing","e005_seasonal_peer","e006_seq_gaps"]:
    df = load_saved(n+".parquet")
    print(n, [c for c in df.columns if c not in ("household_key","snapshot_day")][:20], "...")
print()
v = snapshot()
dem = v.demographics
print(dem.shape)
print(dem.head(3))
for c in dem.columns:
    if c != "household_key":
        print(c, dem[c].value_counts().to_dict())
