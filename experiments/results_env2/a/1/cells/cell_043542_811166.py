
import pandas as pd, numpy as np
from agent_api import load_saved, save_table

# Full diagnostics (redo truncated parts)
oof = load_saved("oof_e5.parquet"); oof["ae"]=(oof.future_spend_4w-oof.pred).abs()
print("OOF MAE by snapshot (incl 403):")
print(oof.groupby("snapshot_day")["ae"].agg(["size","mean"]).round(2).tail(4))

allF = load_saved("allF.parquet")
print("\nallF:", allF.shape, "| val rows:", (allF.snapshot_day>=459).sum(), "| train rows:", (allF.snapshot_day<459).sum())
print("n features:", allF.shape[1]-2)

names = ["e004_preds","e005_preds","e011_preds","e018_preds","e016_preds","e009_preds","e003_preds"]
P = {n: load_saved(n+".parquet").set_index(["household_key","snapshot_day"])["prediction"] for n in names}
C = pd.DataFrame(P).corr()
print("\ncorr among top tables:")
print(C.round(4).to_string())

# E019: equal-weight blend of the 4 best distinct recipes
blend = (P["e005_preds"] + P["e011_preds"] + P["e004_preds"] + P["e018_preds"]) / 4.0
out = blend.reset_index(); out.columns = ["household_key","snapshot_day","prediction"]
p = save_table(out, "e019_blend_preds")
print("\nsaved:", p, out.shape)
print(out["prediction"].describe().round(2))
