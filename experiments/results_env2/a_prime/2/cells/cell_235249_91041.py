import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
tr = v.transactions
tt = agent_api.train_targets()
tgt = tt[tt.snapshot_day==431].set_index("household_key").future_spend_4w

# Correlation of TARGET with its own lag-1 (spend in 28d window ending at snapshot) vs other windows
# Also: how noisy is the target? autocorr of household 28d spend series
E = agent_api.load_saved("e011_discounts.parquet")
E431 = E[E.snapshot_day==431].set_index("household_key")
t431 = tgt.reindex(E431.index)

# ratio target / spend_28: distribution
r = (t431/(E431.spend_28+1))
print("target/spend_28 ratio quantiles:", r.quantile([0.1,0.25,0.5,0.75,0.9]).round(2).to_dict())

# distribution of target by decile of spend_28: how much does model need to shrink?
qs = pd.qcut(E431.spend_28, 10, duplicates="drop")
print("\nmean target by spend_28 decile:")
print(t431.groupby(qs).agg(["mean","count"]).round(1))

# what fraction of target variance explained by spend_28 alone (R2)
X = E431.spend_28.values
r2 = np.corrcoef(X, t431)[0,1]**2
print("\nR2 spend_28 vs target:", round(r2,3))
# and best linear combo of all numeric E011
Xall = E431.drop(columns=["household_key","snapshot_day"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
Xall = np.column_stack([np.ones(len(Xall)), Xall])
b, *_ = np.linalg.lstsq(Xall, t431.values, rcond=None)
pred = Xall@b
print("in-sample R2 all E011 feats at 431:", round(1 - ((t431-pred)**2).sum()/((t431-t431.mean())**2).sum(),3))
print("in-sample MAE:", round(np.abs(t431-pred).mean(),2))
