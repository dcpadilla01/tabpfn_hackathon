
import agent_api, pandas as pd

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
out = e.copy()
for name in ["hazard_v1","cal2_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    nc = [c for c in d.columns if c not in key and c not in out.columns]
    out = out.merge(d[key+nc], on=key, how="left")
print("final shape:", out.shape, "| dups:", out[key].duplicated().sum(), "| dtypes:", set(out.dtypes.astype(str)))
p = agent_api.save_table(out, "e020_final_v1")
print("saved:", p)
