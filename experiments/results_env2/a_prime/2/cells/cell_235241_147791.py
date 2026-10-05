import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

# residual after best linear combo of E011's top spend features at 431
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
base_feats = ["spend_28","spend_84","spend_180","ew_28","ew_84","ew_180","spend_365","lr_mean28","avg_basket_84","baskets_28","baskets_84","trips_per_wk_84","spend_364","rd_84","rd_364"]
X = E431[base_feats].apply(pd.to_numeric, errors="coerce").fillna(0).values
X = np.column_stack([np.ones(len(X)), X])
y = t431.values
bcoef, *_ = np.linalg.lstsq(X, y, rcond=None)
res = y - X@bcoef
print("base MAE at 431:", round(np.abs(res).mean(),3))

# candidate new features on 364d window, corr with residual
w = tr[tr.day >= 459-364].merge(p[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
cands = {}
cands["gas_abs_364"] = gs["KIOSK-GAS"]
cands["gas_share_364"] = gs["KIOSK-GAS"]/gs.sum(axis=1)
b = w.groupby(["household_key","basket_id"]).agg(s=("sales_value","sum"))
cands["max_basket_364"] = b.groupby("household_key")["s"].max()
cands["top2_baskets_364"] = b.sort_values("s", ascending=False).groupby("household_key").head(2).groupby("household_key")["s"].sum()
cands["n_lines_364"] = w.groupby("household_key").size()
cands["n_distinct_products_364"] = w.groupby("household_key")["product_id"].nunique()
cands["units_364"] = w.groupby("household_key")["quantity"].sum()
cands["std_basket_364"] = b.groupby("household_key")["s"].std()
cands["iqr_basket_364"] = b.groupby("household_key")["s"].quantile(0.75)-b.groupby("household_key")["s"].quantile(0.25)
cands["p90_basket_364"] = b.groupby("household_key")["s"].quantile(0.9)

for k, s in cands.items():
    s = s.reindex(E431.index).fillna(0) if k!="std_basket_364" else s.reindex(E431.index)
    cc = np.corrcoef(pd.to_numeric(s, errors="coerce").fillna(0), res)[0,1]
    print(f"{k}: corr_res={cc:.3f}")
