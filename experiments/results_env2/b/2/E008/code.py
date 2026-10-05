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


# ---- cell ----
import agent_api as A
v = A.snapshot(459)
print(type(v))
print([a for a in dir(v) if not a.startswith("_")])
print("households:", type(v.households), v.households if v.households is None else str(v.households)[:200])
print("day:", v.day, "week:", v.week)
tx = v.table("transactions")
print("tx shape", tx.shape, "max day", tx.day.max())
print(A.describe_tables())


# ---- cell ----
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


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np

def season_fn(view, s):
    tx = view.table("transactions")
    first = tx.groupby("household_key")["day"].min()
    need = first[first <= s - 84].index
    t = tx[tx.household_key.isin(need)]
    g = t.groupby("household_key")
    spend_28d = t[t.day > s-28].groupby("household_key")["sales_value"].sum()
    spend_364d = t[t.day > s-364].groupby("household_key")["sales_value"].sum()
    out = pd.DataFrame(index=pd.Index(need, name="household_key"))
    out["spend_28d"] = spend_28d.reindex(out.index).fillna(0.0)
    out["spend_364d"] = spend_364d.reindex(out.index).fillna(0.0)
    for name, lo, hi in [("ya", s-364, s-336), ("l336", s-336, s-308), ("l392", s-392, s-364)]:
        w = t[(t.day > lo) & (t.day <= hi)]
        agg = w.groupby("household_key").agg(sp=("sales_value","sum"), tr=("basket_id","nunique"))
        agg = agg.reindex(out.index).fillna(0.0)
        fd = first.reindex(out.index)
        cov = (np.minimum(hi, s) - np.maximum(lo + 1, fd) + 1).clip(lower=0) / 28.0
        out[name + "_cov"] = cov
        out[name + "_spend"] = agg["sp"].where(cov > 0)
        out[name + "_trips"] = agg["tr"].where(cov > 0)
    out["ya_basket"] = out["ya_spend"] / out["ya_trips"].replace(0, np.nan)
    out["ratio_ya_28"] = out["ya_spend"] / (out["spend_28d"] + 1.0)
    out["ratio_ya_base"] = out["ya_spend"] / (out["spend_364d"] / 13.0 + 1.0)
    return out

bf = A.build_features(season_fn)
print("built", bf.shape)
tt = A.train_targets()
m = tt.merge(bf, on=["household_key","snapshot_day"], how="left")
cov1 = m[m.ya_cov > 0]
print("train rows with ya coverage:", len(cov1), "of", len(m))
print("corr ya_spend:", round(cov1.ya_spend.corr(cov1.future_spend_4w),4),
      "| ratio_ya_28:", round(cov1.ratio_ya_28.corr(cov1.future_spend_4w),4))
print(bf.groupby("snapshot_day").ya_cov.mean().round(2).to_dict())

demo = A.load_saved("demo_v1.parquet")
comb = demo.merge(bf.drop(columns=["spend_28d","spend_364d"]), on=["household_key","snapshot_day"], how="left")
print("combined", comb.shape, "cols", comb.shape[1]-2)
p = A.save_table(comb, "season_demo_v1")
print("saved", p)
