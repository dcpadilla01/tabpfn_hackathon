import pandas as pd, numpy as np
from agent_api import load_saved, save_table
base = load_saved("e016_merged.parquet")
new = load_saved("e017_phase.parquet")
m = base.merge(new, on=["household_key","snapshot_day"], how="inner")
print("merged:", m.shape, "n feat:", m.shape[1]-2)
p = save_table(m, "e017_merged.parquet")
print("saved:", p)
