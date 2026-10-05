
import pandas as pd, numpy as np, agent_api
from scipy.optimize import minimize

feats = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values
s28 = df.spend_28.values

def mae(p, y): return np.mean(np.abs(p-y))

# affine calibration of spend_28 -> y, minimizing MAE
def obj(theta):
    a,b = theta
    return mae(a+b*s28, y)
res = minimize(obj, x0=[np.median(y)-np.median(s28), 1.0], method="Nelder-Mead")
print("affine calib spend_28:", res.x, "MAE:", res.fun)
print("MAE raw spend_28:", mae(s28, y))
print("MAE pred median:", mae(np.full(len(y), np.median(y)), y))

# quantiles of y given zero vs positive
print("zero frac:", (y==0).mean(), "zero frac given s28==0:", (y[s28==0]==0).mean() if (s28==0).sum()>0 else None)

# residual after affine: correlation with other features
a,b = res.x
resid = y - (a+b*s28)
num_cols = [c for c in df.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
cors = df[num_cols].apply(lambda c: np.corrcoef(c.fillna(c.median()), resid)[0,1])
print(cors.sort_values(key=np.abs, ascending=False).head(20))
