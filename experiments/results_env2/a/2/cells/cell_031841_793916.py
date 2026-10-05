import agent_api as A
import pandas as pd, numpy as np

oof = A.load_saved("oof_e016_cv.parquet")
tt = A.train_targets()
print("oof dtypes:", oof.dtypes.to_dict())
print("tt dtypes:", tt.dtypes.to_dict())
print("oof y NaN:", oof.y.isna().sum(), " y describe:", oof.y.describe().round(2).to_dict())

m = oof.merge(tt, on=["household_key","snapshot_day"], how="left", suffixes=("","_tt"))
print("merged rows:", len(m))
m["diff"] = m.y - m.future_spend_4w
print("rows equal (|diff|<1e-6):", (m["diff"].abs()<1e-6).sum(), " NaN tt:", m.future_spend_4w.isna().sum())
print(m[["household_key","snapshot_day","y","future_spend_4w","diff"]].head(10))
print("rows where diff!=0 by day:")
print(m.assign(ne=m["diff"].abs()>1e-6).groupby("snapshot_day")["ne"].agg(["sum","count"]))
# maybe y is a shifted/rolled target? check correlation
print("corr(y, tt):", m[["y","future_spend_4w"]].corr().iloc[0,1].round(4))
