import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
oof = A.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
r = (oof.future_spend_4w - oof.oof_med).values
bins = pd.qcut(oof.oof_med, 10, duplicates="drop").astype(str).values
oof["bin"] = bins
sh = oof.groupby("bin").res_med.median() if "res_med" in oof else None
oof["res_med"] = r
sh = oof.groupby("bin").res_med.median()
adj = r - pd.Series(bins).map(sh).values
print("MAE:", round(np.abs(r).mean(),3), "-> after per-decile median shift:", round(np.abs(adj).mean(),3))
print("\nshifts:", sh.round(2).to_dict())

# per-decile shrink: pred' = a_b * pred + b_b fitted to minimize MAE? try scaling toward median
oof["pred"] = oof.oof_med
oof["y"] = oof.future_spend_4w
sc = oof.groupby("bin").apply(lambda d: pd.Series({
    "a": np.median(d.y)/np.median(d.pred) if np.median(d.pred)>1 else 1.0}), include_groups=False)
print("\nmedian-ratio per bin:", sc.a.round(2).to_dict())
adj2 = oof.pred.values * pd.Series(bins).map(sc.a).values
print("MAE after per-decile median-ratio scaling:", round(np.abs(adj2-oof.y.values).mean(),3))

# stability of shifts across days
print("\nshift by day x bin (top bins):")
piv = oof.pivot_table(index="snapshot_day", columns="bin", values="res_med", aggfunc="median")
print(piv.round(1).to_string())
