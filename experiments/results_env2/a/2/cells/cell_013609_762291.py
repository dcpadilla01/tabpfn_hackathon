import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"])
print(g.round(1))
oof = agent_api.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
res = oof.future_spend_4w - oof.oof_med
print("median resid:", res.median(), "mean resid:", res.mean())
# what shift minimizes MAE of oof_med?
best = min(np.arange(-40,41,2), key=lambda s: np.abs(res-s).mean())
print("best shift on oof:", best, "MAE:", np.abs(res-best).mean(), "vs base:", np.abs(res).mean())
# week mapping
for d in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    print(d, (d+8)//7, end="  | ")
print()
# check week_of_year feature values in feats_v3
fv3 = agent_api.load_saved("feats_v3.parquet")
print(fv3.groupby("snapshot_day").week_of_year.first())
