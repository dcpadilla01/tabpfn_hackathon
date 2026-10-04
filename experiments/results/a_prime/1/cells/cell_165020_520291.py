import numpy as np, pandas as pd
tt = train_targets()
best = load_saved("e019_everything.parquet")
d = best.merge(tt, on=["household_key","snapshot_day"])
y = d["future_spend_4w"].values.astype(float)
# which predictors are best per-snapshot (like a one-feature model)?
# crude: correlation and MAE of prediction = snapshot-day mean of feature (i.e. feature as-is)
num = d.drop(columns=["household_key","snapshot_day","future_spend_4w"])
num = num.select_dtypes(include=[np.number])
res=[]
for c in num.columns:
    v = num[c].values
    m = ~np.isnan(v)
    if m.sum() < len(v)*0.5: continue
    r = np.corrcoef(v[m], y[m])[0,1]
    res.append((c, r))
res.sort(key=lambda t:-abs(t[1]))
print("Top |corr| with future_spend_4w:")
for c,r in res[:35]: print(f"  {c:28s} {r: .3f}")
print()
# correlation among the top spend-level features
top = ["spend_l1","spend_l2","spend_l3","spend_l123_mean","spend_rate28","momentum","spend_7d","spend_14d","nspend28","newm4","newm8","newm13","nwmean12","spend_ly4w","spend_l13","peer_recent28","cohort_prior4w","spend_total","nspend_ly"]
sub = d[top].copy()
print(sub.corrwith(d["future_spend_4w"]).round(3).to_string())
print()
print("corr spend_l1 vs others:")
print(sub.drop(columns=["spend_l1"]).corrwith(sub["spend_l1"]).round(3).to_string())