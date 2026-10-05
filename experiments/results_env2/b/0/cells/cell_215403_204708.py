import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
s28 = val["spend_28"].fillna(0).values
s56 = val["spend_56"].fillna(0).values
s84 = val["spend_84"].fillna(0).values
print("MAE mean-target:", round(float(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean())),2))
print("MAE spend28:", round(float(np.abs(y-s28).mean()),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"MAE {k}*spend28:", round(float(np.abs(y-k*s28).mean()),2))
res=[]
for a in np.linspace(0,1,11):
    p = a*s28 + (1-a)*(s56-s28)
    res.append((round(float(a),2), round(float(np.abs(y-p).mean()),2)))
print("blend sweep (spend28 vs spend56-spend28):", res)

# how different is spend_28 from lag1 (trailing 28 days ending s-1)?
def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    def wsum(lo,hi):
        w = tx[(tx.day>=lo)&(tx.day<=hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["lag1"] = wsum(s-28, s-1)
    out["lag2"] = wsum(s-56, s-29)
    return out.reindex(hh).fillna(0.0)
feats = agent_api.build_features(fn)
m2 = tt.merge(feats, on=["household_key","snapshot_day"])
v2 = m2[m2.snapshot_day==431]
print("\nMAE lag1:", round(float(np.abs(v2.future_spend_4w.values - v2.lag1.values).mean()),2))
print("corr(lag1, spend_28):", round(float(np.corrcoef(v2.lag1, v2.spend_28.fillna(0))[0,1]),4))
print("mean |lag1-spend28|:", round(float((v2.lag1-v2.spend_28.fillna(0)).abs().mean()),2))
