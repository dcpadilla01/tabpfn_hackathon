import numpy as np, pandas as pd
df = load_saved("e019_everything.parquet")   # E017 best table
tt = train_targets()
d = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
print("merged", d.shape)
y = d["future_spend_4w"].values.astype(float)
print("target mean/med", y.mean().round(2), np.median(y))
print("pcts", np.percentile(y,[10,25,50,75,90,95,99]).round(1))
print("zero share", (y==0).mean().round(3))
print("naive MAEs:")
for c in ["spend_l1","spend_l2","spend_l123_mean","spend_rate28","newm4","nwmean12","spend_ly4w","cohort_prior4w"]:
    if c in d: print("  ", c, np.abs(d[c].fillna(0).values - y).mean().round(2))
print("  global median", np.abs(np.median(y)-y).mean().round(2))
# per-snapshot target mean (drift check)
g = d.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median","size"])
print(g.round(1))
# per-snapshot mean of key predictors
for c in ["spend_l1","spend_l13","spend_ly4w"]:
    if c in d: print(c, d.groupby("snapshot_day")[c].mean().round(1).values)