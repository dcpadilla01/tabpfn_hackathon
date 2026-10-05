
import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets()
m = df.merge(tt, on=["household_key","snapshot_day"], how="left")
tr = m[m.future_spend_4w.notna()]; va = m[m.future_spend_4w.isna()]
hh_tr, hh_va = set(tr.household_key), set(va.household_key)
print("households: train", len(hh_tr), "val", len(hh_va), "val in train:", len(hh_va & hh_tr), f"({len(hh_va & hh_tr)/len(hh_va):.1%})")

# persistence: between-household share of target variance (train rows)
g = tr.groupby("household_key").future_spend_4w.agg(["mean","count","std"])
mu = tr.future_spend_4w.mean()
ssb = (g["count"]*(g["mean"]-mu)**2).sum()
sst = ((tr.future_spend_4w-mu)**2).sum()
print(f"between-household variance share: {ssb/sst:.3f}")
print("pooled std:", tr.future_spend_4w.std().round(1), " mean within-household std:", g['std'].mean().round(1))

# how well would a pure 'household mean from train history' predictor do on val?
# use household's mean target over TRAIN snapshots only, predict val rows
hm = tr.groupby("household_key").future_spend_4w.mean()
pred = va.household_key.map(hm).fillna(mu)
print("val MAE of household-mean-from-train predictor:", np.abs(pred - va.future_spend_4w.values if False else 0))  # placeholder
# careful: val targets unknown to us; can't compute. Instead check within-train consistency:
# predict each train row by mean of OTHER train snapshots of same household (leave-one-out)
tmp = tr.merge(tr.groupby("household_key").future_spend_4w.transform("sum").rename("s"), left_index=True, right_index=True)
tmp = tmp.merge(tr.groupby("household_key").future_spend_4w.transform("count").rename("n"), left_index=True, right_index=True)
loo = (tmp.s - tmp.future_spend_4w)/(tmp.n-1)
print("train LOO MAE of household-mean predictor:", np.abs(loo - tr.future_spend_4w).mean().round(2))
print("train MAE of global mean predictor:", np.abs(tr.future_spend_4w - mu).mean().round(2))
