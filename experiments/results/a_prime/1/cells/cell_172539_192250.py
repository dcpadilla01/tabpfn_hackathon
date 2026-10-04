import agent_api as A, pandas as pd, numpy as np
base = A.load_saved("e018_union_full.parquet")
tt = A.train_targets()
tr = base.merge(tt, on=["household_key","snapshot_day"], how="inner")
y = tr.future_spend_4w.values.astype(float)
ns = A.load_saved("nf_seasonal.parquet")[["household_key","snapshot_day","nseas_uplift","nseas_uplift_mean"]]
tu = A.load_saved("e019_true_union.parquet")[["household_key","snapshot_day","coupon_disc_84","match_disc_84"]]
ds = A.load_saved("e017_disc_seasonal.parquet")
dscols = [c for c in ds.columns if c.endswith(("_x","_y"))]
print("ds sample cols:", dscols[:6], "...", len(dscols))
# find the ones that are tenure/nspend_l/nratio_l
cand = {}
for c in dscols:
    root = c[:-2]
    if root in ("tenure","nspend_l","nratio_l"):
        cand[c] = root
print(cand)
m = tr.merge(ns, on=["household_key","snapshot_day"]).merge(tu, on=["household_key","snapshot_day"]).merge(ds[["household_key","snapshot_day"]+list(cand)], on=["household_key","snapshot_day"])
for c in ["nseas_uplift","nseas_uplift_mean","coupon_disc_84","match_disc_84"]+list(cand):
    x = m[c].values.astype(float); ok = np.isfinite(x)
    print(c, "nan%", round(100*(1-ok.mean()),1), "corr", round(np.corrcoef(x[ok],y[ok])[0,1],4) if ok.sum()>100 else "NA")