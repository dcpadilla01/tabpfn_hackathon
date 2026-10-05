
import numpy as np, pandas as pd
from agent_api import load_saved, snapshot, save_table

df = load_saved("rfm_cadence_v1.parquet")
snap = snapshot()
demo = snap.demographics.copy()
dd = pd.get_dummies(demo.drop(columns=["household_key"]).astype(str), prefix_sep="=")
dd.insert(0, "household_key", demo["household_key"].values)
out = df.merge(dd, on="household_key", how="left")
dcol = [c for c in dd.columns if c != "household_key"]
out["has_demo"] = out[dcol].notna().any(axis=1).astype(int)
out[dcol] = out[dcol].fillna(0).astype(int)
print(out.shape)
path = save_table(out, "demo_v1.parquet")
print(path)
