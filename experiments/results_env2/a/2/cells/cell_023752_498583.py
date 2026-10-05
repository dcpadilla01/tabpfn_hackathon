
import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
print(oof.shape, oof.snapshot_day.unique())
print(oof.head())
comps = ["med_v3","med_all","hgbq_v3","hgbq_all"]
for c in comps:
    m = np.abs(oof[c]-oof["y"]).mean()
    print(c, "MAE", round(m,3))

# current E016 blend weights: 0.4 medXGB(v3) + ... need to recall; test some blends
def mae(p): return np.abs(p-oof["y"]).mean()
print("blend .5med_v3+.5hgbq_all:", mae(.5*oof.med_v3+.5*oof.hgbq_all))
print("blend .4med_v3+.3med_all+.3hgbq_all:", mae(.4*oof.med_v3+.3*oof.med_all+.3*oof.hgbq_all))
print("mean of 4:", mae(oof[comps].mean(axis=1)))

# per-snapshot bias
oof["resid"] = oof["y"] - oof["med_all"]
print(oof.groupby("snapshot_day").resid.agg(["mean","median","count"]))

# calibration by prediction bucket
oof["bucket"] = pd.qcut(oof["med_all"], 10, duplicates="drop")
print(oof.groupby("bucket", observed=True).agg(pred_med=("med_all","median"), y_med=("y","median"), y_mean=("y","mean"), n=("y","size")))
