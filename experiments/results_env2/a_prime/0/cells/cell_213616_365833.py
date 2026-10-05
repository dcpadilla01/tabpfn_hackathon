import numpy as np, pandas as pd
v = agent_api.snapshot()  # capped at 459
tx = v.table("transactions")[["household_key","day","sales_value","basket_id"]]

def win_spend(d0, d1):
    m = tx[(tx.day>=d0)&(tx.day<=d1)]
    return m.groupby("household_key").sales_value.sum()

# check: spend at snapshot 431 (window 432..459) vs lag13 window (68..95), lag1 (404..431)
fut = win_spend(432,459)
lag1 = win_spend(404,431); lag13 = win_spend(68,95); lag26 = win_spend(-260,-233)  # lag26 mostly unavailable
hh = fut.index
df = pd.DataFrame({"y":fut}).join(lag1.rename("lag1")).join(lag13.rename("lag13")).fillna(0.0)
df = df[df.index.isin(v.table("transactions").household_key.unique())]
print("n hh:", len(df))
print("corr lag1:", df.y.corr(df.lag1).round(3), " corr lag13:", df.y.corr(df.lag13).round(3))
# partial: does lag13 add beyond lag1?
import numpy as np
A = np.c_[np.ones(len(df)), df.lag1, df.lag13]
b,_,_,_ = np.linalg.lstsq(A, df.y, rcond=None)
pred = A@b
print("lag1-only MAE:", np.abs(df.lag1-df.y).mean().round(2), " lag1+lag13 OLS MAE:", np.abs(pred-df.y).mean().round(2))
# zero structure: P(y=0 | lag1=0)
print("P(y=0|lag1=0):", (df[df.lag1==0].y==0).mean().round(3), " P(y=0|lag1>0):", (df[df.lag1>0].y==0).mean().round(3))
