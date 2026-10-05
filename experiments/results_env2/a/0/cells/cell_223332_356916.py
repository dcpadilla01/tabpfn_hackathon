import agent_api as A, pandas as pd, numpy as np, time
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

f = A.load_saved("feats_v3.parquet")
tt = A.train_targets()
df = tt.merge(f.drop(columns=["index"]), on=["household_key","snapshot_day"], how="left")
print("merged", df.shape, "NaN rows:", df.isna().any(axis=1).sum())

y = df["future_spend_4w"].values
print("target: mean %.1f median %.1f zero-frac %.3f" % (y.mean(), np.median(y), (y==0).mean()))
print("target by snapshot:")
print(df.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median",lambda s:(s==0).mean(),"count"]))

# validation predictions from E005 vs features
p5 = A.load_saved("pred_e005.parquet")
fv = f[f.snapshot_day>=459]
m = fv.merge(p5, on=["household_key","snapshot_day"])
print("pred_e005 stats: mean %.1f median %.1f zero %.3f" % (m.prediction.mean(), m.prediction.median(), (m.prediction==0).mean()))
print("corr pred vs exp4w_blend:", np.corrcoef(m.prediction, m.exp4w_blend)[0,1].round(3))
print("corr pred vs spend_28:", np.corrcoef(m.prediction, m.spend_28)[0,1].round(3))
