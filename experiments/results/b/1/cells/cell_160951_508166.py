import numpy as np, pandas as pd
import agent_api as A

e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
removed = ['ly_spend','ly_ratio','macro_rel','macro_fut_seas','usual13_mfs']
kept = [c for c in base_cols if c not in removed]

# merge e009 (kept) with the e001 columns needed as rank sources
src_cols = ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]
out = e009[["household_key","snapshot_day"]+kept].merge(
    e001[["household_key","snapshot_day"]+src_cols], on=["household_key","snapshot_day"], how="inner")
print("merged:", out.shape)

# within-snapshot percentile ranks
for c in src_cols:
    out["rk_"+c] = out.groupby("snapshot_day")[c].rank(pct=True)
A.save_table(out, "e013_ranks.parquet")
print("saved:", out.shape, "| kept base:", len(kept), "| ranks:", len(src_cols), "| total feats:", out.shape[1]-2)
print(out.head(2).iloc[:, :12].to_string())
