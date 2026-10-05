import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
feat = [c for c in t.columns if c not in ("household_key","snapshot_day")]

val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
spend28 = val["spend_28"].values; s56 = val["spend_56"].values
print("--- pseudo-val snapshot 431, n=", len(val))
print("mean-target MAE:", round(float(np.abs(y - m[m.snapshot_day<431].future_spend_4w.mean())),2))
print("predict spend_28 MAE:", round(float(np.abs(y-spend28).mean()),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"predict {k}*spend_28 MAE:", round(float(np.abs(y-k*spend28).mean()),2))
best=None
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)
    mae = float(np.abs(y-p).mean())
    if best is None or mae<best[1]: best=(round(float(a),2),round(mae,2))
print("best blend spend28 vs (spend56-spend28):", best)

# distribution of target vs spend_28
print("\ntarget mean", round(float(y.mean()),1), "spend28 mean", round(float(spend28.mean()),1))
print("corr(target, spend28):", round(float(np.corrcoef(y,spend28)[0,1]),3))
