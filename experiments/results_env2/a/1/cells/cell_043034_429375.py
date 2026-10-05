
import pandas as pd, numpy as np
from agent_api import load_saved

names = ["e001_preds","e002_preds","e003_preds","e004_preds","e005_preds","e007_preds",
         "e008_preds","e009_preds","e010_preds","e011_preds","e012_preds","e014_preds",
         "e016_preds","e017_preds","e018_preds"]
tabs = {}
for n in names:
    df = load_saved(n + ".parquet")
    tabs[n] = df
    print(n, df.shape, list(df.columns))

base = tabs["e005_preds"][["household_key","snapshot_day"]]
for n, df in tabs.items():
    m = df.merge(base, on=["household_key","snapshot_day"], how="inner")
    print(n, "aligned:", len(m), "/", len(base))

P = pd.DataFrame({n: df.set_index(["household_key","snapshot_day"])["prediction"]
                  for n, df in tabs.items()})
print("\ncorr matrix:")
print(P.corr().round(3))
print("\nsummary:")
print(P.describe().T[["mean","std","min","max"]].round(2))

oof = load_saved("oof_e5.parquet")
print("\noof_e5:", oof.shape, list(oof.columns))
print(oof.head(3))
rep = load_saved("repro_e5.parquet")
print("repro_e5:", rep.shape, list(rep.columns))
print(rep.head(3))
