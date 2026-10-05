import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w
common = tgt.index.intersection(tr.household_key.unique())

# KIOSK-GAS spend share and absolute
w = tr[tr.day >= 459-364].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
gas_abs = gs.get("KIOSK-GAS", pd.Series(0, index=gs.index))
gas_share = gas_abs/gs.sum(axis=1)
print("corr gas_abs:", round(np.corrcoef(gas_abs.loc[common], tgt.loc[common])[0,1],3))
print("corr gas_share:", round(np.corrcoef(gas_share.loc[common], tgt.loc[common])[0,1],3))

# spend in trailing 7d, 14d — already in E011? E011 has spend_7, spend_14. Check corr of E011 features with target at 431
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
corrs = E431.drop(columns=["household_key","snapshot_day"]).corrwith(t431).sort_values()
print("\nlowest corr E011 feats:")
print(corrs.head(8).round(3))
print("highest corr E011 feats:")
print(corrs.tail(12).round(3))

# how much signal in spend_28 alone? residual analysis
import numpy.linalg as la
X = E431[["spend_28"]].fillna(0).values
y = t431.values
b = la.lstsq(X, y, rcond=None)[0]
res = y - X@b
print("\nMAE of spend_28*coef at 431:", round(np.abs(res).mean(),2), "coef:", round(b[0],3))
print("MAE of median:", round(np.abs(y-np.median(y)).mean(),2))
