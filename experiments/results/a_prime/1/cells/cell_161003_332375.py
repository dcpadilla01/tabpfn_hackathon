import agent_api as A, pandas as pd, numpy as np

e10 = A.load_saved("e010_l13fix.parquet")
e11 = A.load_saved("e011_demo.parquet")
e03 = A.load_saved("e003_catmix.parquet")
print("e010 cols not in e003:", [c for c in e10.columns if c not in e03.columns])
print("e011 cols not in e003:", [c for c in e11.columns if c not in e03.columns])
print("e010 shape", e10.shape, "e011 shape", e11.shape)
# check l13 NaN pattern in e010 vs e011
m = e11[["household_key","snapshot_day","spend_l13","tenure"]].merge(
    e10[["household_key","snapshot_day","spend_l13","has_real_l13","l13_over_recent"]],
    on=["household_key","snapshot_day"], suffixes=("_e11","_e10"))
print("e11 l13 NaN frac:", m["spend_l13_e11"].isna().mean().round(3), " e10:", m["spend_l13_e10"].isna().mean().round(3))
print(m[["snapshot_day","tenure","has_real_l13"]].drop_duplicates("snapshot_day").groupby("snapshot_day")["has_real_l13"].mean().round(2))

# target distribution / zero mass
tt = A.train_targets()
print("\ntarget describe:"); print(tt["future_spend_4w"].describe().round(2))
print("zero frac:", (tt["future_spend_4w"]==0).mean().round(3))
