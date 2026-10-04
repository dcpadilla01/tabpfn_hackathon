import numpy as np, pandas as pd
v = snapshot()
camp = v.campaigns; ct = v.campaign_targets
# campaign 8 = TypeA 412-460; campaign 9 = TypeB 435-467. Who is targeted?
t8 = ct[ct.campaign==8].household_key.unique()
t9 = ct[ct.campaign==9].household_key.unique()
print("campaign8 targets:", len(t8), "campaign9 targets:", len(t9))
print("overlap:", len(set(t8)&set(t9)))
# do targeted households actually spend more?
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
d["t8"] = d.household_key.isin(t8).astype(int)
d["t9"] = d.household_key.isin(t9).astype(int)
print(d.groupby("t8")["future_spend_4w"].agg(["mean","median","size"]))
print(d.groupby("t9")["future_spend_4w"].agg(["mean","median","size"]))
# TypeA targets across time - is targeting persistent per household?
tA = ct[ct.description=="TypeA"].groupby("household_key")["campaign"].nunique()
print("TypeA: hh with multiple targeted campaigns:", (tA>1).sum(), "of", len(tA))
tB = ct[ct.description=="TypeB"].groupby("household_key")["campaign"].nunique()
print("TypeB:", (tB>1).sum(), "of", len(tB))
tC = ct[ct.description=="TypeC"].groupby("household_key")["campaign"].nunique()
print("TypeC:", (tC>1).sum(), "of", len(tC))