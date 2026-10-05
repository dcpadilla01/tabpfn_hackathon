import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

t = agent_api.load_saved("e006_zero_inflation.parquet")
tt = agent_api.train_targets()
m = tt.merge(t, on=["household_key","snapshot_day"])
val = m[m.snapshot_day==431]
y = val.future_spend_4w.values
print("y shape", y.shape, "val shape", val.shape)
mu = m[m.snapshot_day<431].future_spend_4w.mean()
print("mu:", mu, type(mu))
d = np.abs(y - mu)
print("d shape", d.shape, "mae", d.mean())
print("MAE mean-target:", round(d.mean(),2))
spend28 = val["spend_28"].values
print("MAE spend28:", round(np.abs(y-spend28).mean(),2))
for k in [0.7,0.8,0.9,1.0,1.1]:
    print(f"MAE {k}*spend28:", round(np.abs(y-k*spend28).mean(),2))
s56 = val["spend_56"].values
res=[]
for a in np.linspace(0,1,21):
    p = a*spend28 + (1-a)*(s56-spend28)
    res.append((round(float(a),2), round(float(np.abs(y-p).mean()),2)))
print("blend sweep:", res)
print("corr(y, spend28):", round(float(np.corrcoef(y,spend28)[0,1]),3))
