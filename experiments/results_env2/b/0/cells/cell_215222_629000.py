import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]
print(m[feat].dtypes.value_counts())
str_cols = [c for c in feat if m[c].dtype == object]
print("string cols:", str_cols)
if str_cols:
    for c in str_cols:
        print(c, m[c].unique()[:8])

# naive baselines on last train snapshot as pseudo-val
val = m[m.snapshot_day==431]
spend28 = val["spend_28"].values; y = val.future_spend_4w.values
print("\n--- pseudo-val snapshot 431, n=", len(val))
print("mean-target MAE:", round(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean()),2))
print("predict spend_28 MAE:", round(np.abs(y-spend28).mean(),2))
for k in [0.7,0.8,0.9,1.0]:
    print(f"predict {k}*spend_28 MAE:", round(np.abs(y-k*spend28).mean(),2))
# blend with spend_56
s56 = val["spend_56"].values
best=None
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)  # spend_56 minus spend_28 = previous block? approx
    mae = np.abs(y-p).mean()
    if best is None or mae<best[1]: best=(a,mae)
print("best blend spend28 vs (spend56-spend28):", best)
