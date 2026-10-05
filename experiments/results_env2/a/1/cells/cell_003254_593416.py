
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

def fitq(**kw):
    b = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
    b.update(kw)
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **b)
    m.fit(Xtr, ytr, sample_weight=w)
    return m

t0=time.time()
m8 = fitq(max_depth=8, min_child_weight=20)
p8 = pd.Series(m8.predict(Xe), index=Xe.index).clip(lower=0)
print("depth8 MAE431 %.2f (%.0fs)" % (np.abs(p8-ye).mean(), time.time()-t0)); t0=time.time()

m36 = fitq(n_estimators=3600, learning_rate=0.02)
p36 = pd.Series(m36.predict(Xe), index=Xe.index).clip(lower=0)
print("3600@.02 MAE431 %.2f (%.0fs)" % (np.abs(p36-ye).mean(), time.time()-t0))

# clip caps on the depth-6 base prediction (recompute quickly with saved-style model? reuse p36/p8)
base = pd.Series(fitq().predict(Xe), index=Xe.index).clip(lower=0)
print("base MAE431 %.2f" % np.abs(base-ye).mean())
for cap in [500, 700, 900, 1100]:
    print("clip@%d -> %.2f" % (cap, np.abs(base.clip(upper=cap)-ye).mean()))
# blend with persistence
for al in [0.1, 0.2, 0.3]:
    per = Xe["seq_mean"] if "seq_mean" in Xe else Xe["spend_84"]
    bl = (1-al)*base + al*per.values
    print("blend seq_mean a=%.1f -> %.2f" % (al, np.abs(bl-ye).mean()))
# 2-seed ensemble
m_s2 = fitq(subsample=0.7, colsample_bytree=0.6)
p_s2 = pd.Series(m_s2.predict(Xe), index=Xe.index).clip(lower=0)
print("2-seed avg MAE431 %.2f" % np.abs(((base+p_s2)/2)-ye).mean())
