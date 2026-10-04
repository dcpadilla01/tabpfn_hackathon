import pandas as pd, numpy as np, agent_api

def L(nm):
    for c in (nm, nm+".parquet"):
        try: return agent_api.load_saved(c)
        except Exception: pass
    raise RuntimeError(nm)

base = L("e011_pruned"); haz = L("e017_hazard_only"); dec = L("e016_dec_predictors")
bcols = set(base.columns)
haz_new = [c for c in haz.columns if c not in bcols and c not in ("household_key","snapshot_day")]
dec_new = [c for c in dec.columns if c not in bcols and c not in ("household_key","snapshot_day")]
print("haz_new:", haz_new)
print("dec_new:", dec_new)

m = base.merge(haz[["household_key","snapshot_day"]+haz_new], on=["household_key","snapshot_day"], how="left")
m = m.merge(dec[["household_key","snapshot_day"]+dec_new], on=["household_key","snapshot_day"], how="left")
# drop the 7 redundant lg2_* cols added in the previous build
m = m.drop(columns=[c for c in m.columns if c.startswith("lg2_")])
feats = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("final", m.shape, "n_feats", len(feats))
print("dups:", m.columns[m.columns.duplicated()].tolist())
p = agent_api.save_table(m, "e019_curated2")
print("saved", p)
