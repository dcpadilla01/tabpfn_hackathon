import pandas as pd, numpy as np
e3 = agent_api.load_saved("e003_union.parquet")
e1 = agent_api.load_saved("e001_rfm.parquet")
print("e3", e3.shape, "e1", e1.shape)
print("E3 COLS:", sorted(map(str, e3.columns)))
t = agent_api.train_targets()
y = t["future_spend_4w"]
print("\nTARGET describe:\n", y.describe())
print("zero share %.3f median %.1f" % ((y==0).mean(), y.median()))
m = t.merge(e3, on=["household_key","snapshot_day"], how="left")
feat = [c for c in e3.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(m[c])]
cat = [c for c in feat if c not in num]
cor = m[num].corrwith(m["future_spend_4w"])
print("\nTOP |corr|:\n", cor.reindex(cor.abs().sort_values(ascending=False).index).head(25))
print("\nNEAR-ZERO CORR:\n", cor.reindex(cor.abs().sort_values().index).head(15))
print("\nCAT COLS:", cat)
print("\nsnapshot_days:", agent_api.snapshot_days())
