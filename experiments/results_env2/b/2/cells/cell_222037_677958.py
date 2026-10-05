import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()

def season_feats(view, s):
    tx = view.table("transactions")
    lo, hi = s-364, s-336   # year-ago future window (lo, hi]
    w = tx[(tx.day > lo) & (tx.day <= hi)]
    g = w.groupby("household_key").agg(ya_spend=("sales_value","sum"),
                                       ya_trips=("basket_id","nunique"))
    first = tx.groupby("household_key")["day"].min()
    f = g.join(first.rename("first_day"))
    f["ya_spend"] = f["ya_spend"].fillna(0.0)
    f["ya_trips"] = f["ya_trips"].fillna(0.0)
    cov = max(0, hi - max(lo, 0))
    f["ya_cov"] = cov / 28.0
    f["tenure"] = s - f["first_day"]
    wk = ((s + 8)//7 - 1) % 52 + 1
    f["week52"] = wk
    f["snapshot_day"] = s
    return f.reset_index()

days = A.snapshot_days()["train"]
parts = []
for s in days:
    v = A.snapshot(s)
    parts.append(season_feats(v, s))
sf = pd.concat(parts, ignore_index=True)
m = tt.merge(sf, on=["household_key","snapshot_day"], how="left")
print("merged", m.shape, "missing", m.ya_spend.isna().sum())
for c in ["ya_spend","ya_trips","tenure","week52"]:
    print(c, "corr", round(m[c].astype(float).corr(m.future_spend_4w), 4))
sub = m[m.ya_cov==1]
print("ya_spend corr (cov=1, n=%d):" % len(sub), round(sub.ya_spend.corr(sub.future_spend_4w),4))
print(m.groupby("week52").future_spend_4w.agg(["mean","count"]).round(1))
