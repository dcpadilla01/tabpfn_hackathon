
import pandas as pd, numpy as np
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in c5 if c in F.columns]
F = F.merge(f5n.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
mq = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.03,
                  n_estimators=2400, max_depth=6, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
mq.fit(Xtr, ytr, sample_weight=w)

Xc, yc = mat([403]); Xe, ye = mat([431])
pc = pd.Series(mq.predict(Xc), index=Xc.index).clip(lower=0)
pe = pd.Series(mq.predict(Xe), index=Xe.index).clip(lower=0)
print("MAE 403: %.2f  MAE 431: %.2f" % (np.abs(pc-yc).mean(), np.abs(pe-ye).mean()))

# residual bias by decile of p (on 403, the calibration snapshot)
d = pd.DataFrame({"p":pc, "y":yc})
d["b"] = pd.qcut(d.p, 10, duplicates="drop")
print(d.groupby("b", observed=True).agg(n=("y","size"), p=("p","mean"), y=("y","mean"),
      bias=("y","mean")).assign(bias=lambda x: x.y - x.p).round(1).to_string())

# recalibrate: quantile-regression of y on p using 403, apply to 431
cal = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.05,
                   n_estimators=300, max_depth=3, min_child_weight=20, subsample=0.9,
                   colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist", n_jobs=8)
cal.fit(pc.to_frame("p"), yc)
pr = pd.Series(cal.predict(pe.to_frame("p")), index=pe.index).clip(lower=0)
print("recal MAE 431: %.2f (raw %.2f)" % (np.abs(pr-ye).mean(), np.abs(pe-ye).mean()))

# global multiplicative / additive checks on 431
for c in [0.95, 1.0, 1.05, 1.1, 1.15]:
    print("scale %.2f -> MAE %.2f" % (c, np.abs(pe*c-ye).mean()))
med = float(yc.median()); print("median y 403: %.2f" % med)
for a in [-5, 0, 5]:
    print("shift %+d -> MAE %.2f" % (a, np.abs(pe+a-ye).mean()))
