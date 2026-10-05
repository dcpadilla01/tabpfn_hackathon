
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key", "snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key", "snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key", "snapshot_day")]
tt = train_targets().set_index(["household_key", "snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key", "snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95, 123, 151, 179, 207, 235, 263, 291, 319, 347, 375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

def fitq(**kw):
    b = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
    b.update(kw)
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **b)
    m.fit(Xtr, ytr, sample_weight=w)
    return m

m = fitq()
pc = pd.Series(m.predict(Xc), index=Xc.index).clip(lower=0)
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
per = Xe["seq_mean"]
d = pd.DataFrame({"p": pe, "per": per.values, "y": ye})

for thr in [30, 50, 70, 100]:
    for al in [0.3, 0.5]:
        weak = d.p < thr
        bl = d.p.copy(); bl[weak] = (1 - al) * d.p[weak] + al * d.per[weak]
        print("thr %3d a=%.1f -> %.2f (n_weak %d)" % (thr, al, np.abs(bl - ye).mean(), weak.sum()))

weak = d.p < 50
print("MAE per on weak rows: %.2f vs model %.2f" % (np.abs(d.per[weak] - d.y[weak]).mean(), np.abs(d.p[weak] - d.y[weak]).mean()))
