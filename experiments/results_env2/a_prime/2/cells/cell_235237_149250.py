import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
common = tgt.index.intersection(tr.household_key.unique())

# basket-level aggregation: max basket, top-2 basket share, basket count
b = tr[tr.day >= 459-364].groupby(["household_key","basket_id"]).agg(s=("sales_value","sum"), d=("day","first"))
per_hh = b.groupby("household_key")["s"]
max_b = per_hh.max(); top2 = b.sort_values("s", ascending=False).groupby("household_key").head(2).groupby("household_key")["s"].sum()
tot = per_hh.sum()
common2 = tgt.index.intersection(tot.index)
print("corr max_basket:", round(np.corrcoef(max_b.loc[common2], tgt.loc[common2])[0,1],3))
print("corr top2_baskets:", round(np.corrcoef(top2.loc[common2], tgt.loc[common2])[0,1],3))
print("corr top2_share:", round(np.corrcoef((top2/tot).loc[common2], tgt.loc[common2])[0,1],3))

# trans_time: weekend share of spend (day%7 in {4,5})
w = tr[tr.day >= 459-364].copy()
w["wknd"] = (w.day % 7).isin([4,5]).astype(int)
wknd_spend = w.groupby("household_key").apply(lambda g: (g.sales_value*g.wknd).sum()/g.sales_value.sum(), include_groups=False)
print("corr weekend spend share:", round(np.corrcoef(wknd_spend.loc[common2], tgt.loc[common2])[0,1],3))

# hour: evening share (trans_time>=1700)
w["eve"] = (w.trans_time>=1700).astype(int)
eve_spend = w.groupby("household_key").apply(lambda g: (g.sales_value*g.eve).sum()/g.sales_value.sum(), include_groups=False)
print("corr evening spend share:", round(np.corrcoef(eve_spend.loc[common2], tgt.loc[common2])[0,1],3))

# gas absolute spend corr at 28d window too
gas28 = tr[tr.day >= 459-28].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
g28 = gas28[gas28.department=="KIOSK-GAS"].groupby("household_key")["sales_value"].sum()
g28 = g28.reindex(tgt.index).fillna(0)
print("corr gas_spend_28 vs target:", round(np.corrcoef(g28, tgt)[0,1],3))
