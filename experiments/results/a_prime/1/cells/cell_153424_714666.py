import agent_api as A
import pandas as pd, numpy as np

key = ["household_key","snapshot_day"]
want = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12','r_zerow13']
src = {}
for nm in ["e003_catmix","nf_p1","nf_transforms","nf_candidates","e004_mkt_full","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    have = [c for c in want if c in df.columns and c not in src]
    if have:
        src[nm] = df[key+have]
        print(nm, "->", have)
missing = [c for c in want if c not in pd.concat([s.drop(columns=key) for s in src.values()], axis=1).columns] if False else []
out = None
for nm, df in src.items():
    out = df if out is None else out.merge(df, on=key, how="inner")
out = out[key + want]
print("compact table:", out.shape, "cols ok:", set(out.columns)==set(key+want))
p = A.save_table(out, "nf_compact19.parquet")
print("saved:", p)
print(out[want].describe().T[["mean","std"]].round(2))