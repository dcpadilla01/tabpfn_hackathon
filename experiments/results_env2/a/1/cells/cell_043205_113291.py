
import pandas as pd, numpy as np
from agent_api import load_saved

oof = load_saved("oof_e5.parquet")
oof["resid"] = oof["future_spend_4w"] - oof["pred"]
oof["abs_err"] = oof["resid"].abs()
print("OOF rows:", len(oof), " OOF MAE: %.3f" % oof["abs_err"].mean(),
      " mean pred %.2f mean target %.2f" % (oof["pred"].mean(), oof["future_spend_4w"].mean()))
g = oof.groupby("snapshot_day").agg(n=("resid","size"), mae=("abs_err","mean"),
        bias=("resid","mean"), mp=("pred","mean"), mt=("future_spend_4w","mean"))
print(g.round(2))

# bias trend vs snapshot day (linear fit)
X = np.vstack([np.ones(len(oof)), oof["snapshot_day"]]).T
y = oof["resid"].values
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
print("resid ~ const %.3f + %.4f*day" % (beta[0], beta[1]))
# weighted by decay toward validation days (459..543): weight ~ 0.5**((459-day)/140)
w = 0.5 ** (np.maximum(0, 459 - oof["snapshot_day"]) / 140.0)
Xw = X * w[:,None]
bw, *_ = np.linalg.lstsq(Xw, y*w, rcond=None)
print("weighted: const %.3f + %.4f*day -> correction at day 459: %.2f, 543: %.2f" %
      (bw[0], bw[1], bw[0]+bw[1]*459, bw[0]+bw[1]*543))

allF = load_saved("allF.parquet")
print("\nallF:", allF.shape)
print([c for c in allF.columns][:60])
