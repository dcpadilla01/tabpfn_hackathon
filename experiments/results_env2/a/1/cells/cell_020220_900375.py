import pandas as pd, numpy as np
from agent_api import load_saved

names = ["e001_preds","e002_preds","e003_preds","e004_preds","e005_preds","e007_preds","e008_preds","e009_preds","e010_preds","e011_preds"]
P = {}
for n in names:
    df = load_saved(n + ".parquet")
    P[n] = df
    print(n, df.shape, list(df.columns))

base = P["e005_preds"][["household_key","snapshot_day"]].copy()
print("base rows", len(base), "unique hh", base.household_key.nunique(), "days", sorted(base.snapshot_day.unique()))

M = base.copy()
for n, df in P.items():
    d = df[["household_key","snapshot_day","prediction"]].rename(columns={"prediction": n})
    M = M.merge(d, on=["household_key","snapshot_day"], how="left")
cols = names
print("NaNs:", M[cols].isna().sum().to_dict())
print(M[cols].describe().loc[["mean","std","min","max"]].round(2))
print(M[cols].corr().round(3))

for n in ["repro_e5","e005_newfeats","e004_new","allF","lagfeats","lagfeats2","f_weekly","e004_features","e002_features"]:
    try:
        df = load_saved(n + ".parquet")
        print("---", n, df.shape, list(df.columns)[:30])
    except Exception as e:
        print(n, "ERR", type(e).__name__, str(e)[:80])
