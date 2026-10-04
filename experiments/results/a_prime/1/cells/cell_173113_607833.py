import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
rb = A.load_saved("nf_robust.parquet")
newc = ["s_day","k_sin1","k_cos1","k_sin2","k_cos2","r_autocorr","r_dsl","r_gap_mean","r_gap_ratio","g_mean_l1"]
newc = [c for c in newc if c not in base.columns]
m = base.merge(rb[["household_key","snapshot_day"]+newc], on=["household_key","snapshot_day"], how="left")
print(m.shape, "added:", newc)
print(m[newc].isna().mean().round(3))
print("s_day==snapshot_day:", (m.s_day==m.snapshot_day).mean())
path = A.save_table(m, "e020_final")
print(path)