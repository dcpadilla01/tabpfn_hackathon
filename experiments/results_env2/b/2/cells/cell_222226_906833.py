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
