import agent_api as A
import pandas as pd, numpy as np

key = ["household_key","snapshot_day"]
want = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12','r_zerow13']
order = ["e003_catmix","nf_p1","nf_transforms","nf_candidates","e004_mkt_full","nf_robust"]
tabs = {nm: A.load_saved(nm+".parquet") for nm in order}
out = None
for nm in order:
    df = tabs[nm]
    have = [c for c in want if c in df.columns and c not in (set(out.columns) if out is not None else set())]
    if not have: continue
    part = df[key+have].copy()
    out = part if out is None else out.merge(part, on=key, how="inner")
print("cols:", sorted(out.columns) == sorted(key+want), out.shape)
out = out[key+want]
p = A.save_table(out, "nf_compact19.parquet")
print("saved:", p)
print(out.isna().mean().round(3).sort_values(ascending=False).head(5))