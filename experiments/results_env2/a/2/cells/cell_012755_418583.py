import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
tt = agent_api.train_targets()
df = oof.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values; med = df.oof_med.values
# calibration: bin pred, compare pred vs median/mean target in bin
bins = np.quantile(med, np.arange(0,1.01,0.1))
df["pb"] = pd.cut(med, bins, include_lowest=True, duplicates="drop")
cal = df.groupby("pb", observed=True).apply(lambda g: pd.Series({
    "n":len(g), "pred_med":g.oof_med.median(), "y_med":g.future_spend_4w.median(),
    "y_mean":g.future_spend_4w.mean(), "mae":np.abs(g.oof_med-g.future_spend_4w).mean()}), include_groups=False)
print(cal.round(2))
# capping test
for cap in [400,600,800,1000,1200,1500,1e9]:
    print("cap",cap, round(np.abs(np.minimum(med,cap)-y).mean(),3))
print("y quantiles:", np.quantile(y,[0.9,0.95,0.99,0.999]).round(1), "max", y.max())
print("pred quantiles:", np.quantile(med,[0.9,0.95,0.99,0.999]).round(1), "max", med.max())
