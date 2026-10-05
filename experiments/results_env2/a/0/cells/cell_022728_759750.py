
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e013.parquet")
y = oof.y.values; p = oof.oof.values

# 1) per-snapshot multiplicative calibration (leave-one-out style: fit on other snapshots)
oof["resid"] = y - p
for s in sorted(oof.snapshot_day.unique()):
    tr = oof[oof.snapshot_day != s]
    a = tr.y.sum()/tr.oof.sum()
    va = oof[oof.snapshot_day == s]
    mae = np.abs(va.oof*a - va.y).mean()
    print("snap %3d: out-scale %.3f  MAE %.3f -> %.3f" % (s, a, np.abs(va.oof-va.y).mean(), mae))

# 2) global isotonic on oof
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip", y_min=0)
iso.fit(p, y)
print("\nisotonic global MAE %.3f" % np.abs(iso.predict(p)-y).mean())
print("iso mapping:", np.round(np.quantile(p,[0,.1,.25,.5,.75,.9,.99]),1),
      "->", np.round(iso.predict(np.quantile(p,[0,.1,.25,.5,.75,.9,.99])),1))

# 3) residual vs prediction size (slope of y on p)
lo = pd.qcut(p, 10, duplicates="drop")
print("\nbin means: p, y, mean resid")
print(oof.groupby(lo, observed=True).agg(p=("oof","mean"), y=("y","mean"), n=("y","size")).round(1))
