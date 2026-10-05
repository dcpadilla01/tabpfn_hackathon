import numpy as np, pandas as pd, agent_api as A, warnings
warnings.filterwarnings("ignore")
df = A.load_saved("e005_marketing.parquet")
aw = A.load_saved("aw_hist.parquet")
m = df.merge(aw, on=["household_key","snapshot_day"], how="left")
awc = [f"aw_{k}" for k in range(1,9)]
m["aw_med8"] = m[awc].median(1)
m["aw_mean8"] = m[awc].mean(1)
key = ["spend_4w","spend_8w","spend_12w","spend_28w","spend_56w","spend_112w","spend_total","nbask_4w","nbask_8w","nbask_28w","nbask_56w","nbask_112w","spend_per_day_total","spend_per_week_112","spend_112_mean4","spend_112_max4","spend_112_min4","spend_112_std","spend_4w_lag1","spend_4w_lag2","spend_4w_lag3"]
for c in key:
    m["lg_"+c] = np.log1p(np.clip(m[c].fillna(0).values,0,None))
drop = ["day","week","snap_day","snap_week","index"]
out = m.drop(columns=[c for c in drop if c in m.columns])
print("cols:", out.shape[1], "rows:", len(out))
print("null rows in key cols:", int(out[["aw_med8","aw_mean8"]].isna().any(1).sum()))
path = A.save_table(out, "e006_awagg")
print(path)
