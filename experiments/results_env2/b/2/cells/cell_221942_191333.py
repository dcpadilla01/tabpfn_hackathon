import agent_api as A, pandas as pd, numpy as np

tt = A.train_targets()
y = tt.future_spend_4w
print("rows", len(tt), "mean %.2f std %.2f zero%% %.3f" % (y.mean(), y.std(), (y==0).mean()))
print("quantiles", y.quantile([.5,.75,.9,.95,.99]).round(1).to_dict())

rfm = A.load_saved("rfm_cadence_v1.parquet")
demo = A.load_saved("demo_v1.parquet")
print("rfm", rfm.shape); print(rfm.columns.tolist())
print("demo", demo.shape); print(demo.columns.tolist())

def season_feats(view, s):
    tx = view.table("transactions")
    hh = view.households
    if hasattr(hh, "columns"):
        idx = pd.Index(hh["household_key"].values, name="household_key")
    else:
        idx = pd.Index(list(hh), name="household_key")
    lo, hi = s-364, s-336   # year-ago future window (lo, hi]
    w = tx[(tx.day > lo) & (tx.day <= hi)]
    g = w.groupby("household_key").agg(ya_spend=("sales_value","sum"),
                                       ya_trips=("basket_id","nunique"))
    first = tx.groupby("household_key")["day"].min()
    f = pd.DataFrame(index=idx).join(g).join(first.rename("first_day"))
    f["ya_spend"] = f["ya_spend"].fillna(0.0)
    f["ya_trips"] = f["ya_trips"].fillna(0.0)
    cov = max(0, hi - max(lo, 0))
    f["ya_cov"] = cov / 28.0
    f["tenure"] = s - f["first_day"]
    wk = ((s + 8)//7 - 1) % 52 + 1
    f["week52"] = wk
    f["sin1"] = np.sin(2*np.pi*wk/52.0); f["cos1"] = np.cos(2*np.pi*wk/52.0)
    f["sin2"] = np.sin(4*np.pi*wk/52.0); f["cos2"] = np.cos(4*np.pi*wk/52.0)
    f["snapshot_day"] = s
    return f.reset_index()

days = A.snapshot_days()["train"]
parts = []
for s in days:
    v = A.snapshot(s)
    parts.append(season_feats(v, s))
sf = pd.concat(parts, ignore_index=True)
m = tt.merge(sf, on=["household_key","snapshot_day"], how="left")
print("merged", m.shape)
for c in ["ya_spend","ya_trips","tenure","week52"]:
    print(c, "corr", round(m[c].astype(float).corr(m.future_spend_4w), 4))
sub = m[m.ya_cov==1]
print("ya_spend corr (cov=1, n=%d):" % len(sub), round(sub.ya_spend.corr(sub.future_spend_4w),4))
print(m.groupby("week52").future_spend_4w.agg(["mean","count"]).round(1))
