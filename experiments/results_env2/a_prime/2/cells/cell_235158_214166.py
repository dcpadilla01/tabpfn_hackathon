import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
p = v.table("products")

# department spend shares over 364d, correlation with target
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
print("n tgt:", len(tgt))

w = tr[tr.day >= 459-364]
w = w.merge(p[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
tot = gs.sum(axis=1)
shares = gs.div(tot, axis=0)

# target correlations
rows=[]
for c in shares.columns:
    common = tgt.index.intersection(shares.index)
    r = np.corrcoef(shares.loc[common, c], tgt.loc[common])[0,1]
    rows.append((c, round(r,3), round(shares[c].mean(),3)))
rows.sort(key=lambda x: -abs(x[1]))
print("dept share corr with target (364d window):")
for r in rows[:15]: print(r)

# store loyalty: main-store share of trips
st = tr[tr.day >= 459-364].groupby(["household_key","store_id"]).size()
main_share = st.groupby(level=0).max()/st.groupby(level=0).sum()
common = tgt.index.intersection(main_share.index)
print("\ncorr main-store trip share vs target:", round(np.corrcoef(main_share.loc[common], tgt.loc[common])[0,1],3))
