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
drop_junk = {"index","d28_ ","sh112_ "} | {c for c in haz_new+dec_new if c.startswith(("tlag_16","tlag_17","tlag_18","d28_ELECT","d28_HOUSEWARES","d28_MEAT-WHSE","d28_PHARMACY","d28_POSTAL","d28_VIDEO","sh112_ELECT","sh112_MEAT-WHSE","sh112_PHARMACY","sh112_VIDEO"))}
haz_new = [c for c in haz_new if c not in drop_junk]
dec_new = [c for c in dec_new if c not in drop_junk]
print("haz_new:", haz_new); print("dec_new:", dec_new)

m = base.merge(haz[["household_key","snapshot_day"]+haz_new], on=["household_key","snapshot_day"], how="left")
m = m.merge(dec[["household_key","snapshot_day"]+dec_new], on=["household_key","snapshot_day"], how="left")
feats = [c for c in m.columns if c not in ("household_key","snapshot_day")]
print("final", m.shape, "n_feats", len(feats))
p = agent_api.save_table(m, "e019_curated3")
print("saved", p)
