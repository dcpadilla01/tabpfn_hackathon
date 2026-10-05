import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

w = tr[tr.day >= 459-364].merge(v.table("products")[["product_id","department"]], on="product_id", how="left")
gs = w.groupby(["household_key","department"])["sales_value"].sum().unstack().fillna(0)
gas_abs = gs["KIOSK-GAS"] if "KIOSK-GAS" in gs.columns else pd.Series(0, index=gs.index)
gas_share = gas_abs/gs.sum(axis=1)
common = tgt.index.intersection(gs.index)
print("corr gas_abs:", round(np.corrcoef(gas_abs.loc[common], tgt.loc[common])[0,1],3))
print("corr gas_share:", round(np.corrcoef(gas_share.loc[common], tgt.loc[common])[0,1],3))

E = agent_api.load_saved("e011_discounts.parquet")
print("E011 snap days:", sorted(E.snapshot_day.unique()))
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)
feats = [c for c in E431.columns if c not in ("household_key","snapshot_day")]
corrs = E431[feats].apply(lambda s: pd.to_numeric(s, errors="coerce")).corrwith(t431).sort_values()
print("\nlowest corr E011 feats:"); print(corrs.head(8).round(3))
print("highest corr E011 feats:"); print(corrs.tail(12).round(3))
