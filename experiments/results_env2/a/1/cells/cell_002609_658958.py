
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

base = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
preds = {}
t0=time.time()
m1 = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **base)
m1.fit(Xtr, ytr, sample_weight=w); preds["quant"] = m1.predict
print("quant %.0fs" % (time.time()-t0)); t0=time.time()

b2 = dict(base); b2["n_estimators"]=1200
m2 = XGBRegressor(objective="reg:absoluteerror", **b2)
m2.fit(Xtr, ytr, sample_weight=w); preds["l1"] = m2.predict
print("l1 %.0fs" % (time.time()-t0)); t0=time.time()

m3 = XGBRegressor(objective="reg:squarederror", **b2)
m3.fit(Xtr, np.log1p(ytr), sample_weight=w)
preds["log"] = lambda X: np.expm1(m3.predict(X))
print("log %.0fs" % (time.time()-t0))

P = {k: pd.Series(np.clip(f(Xe),0,None), index=Xe.index) for k,f in preds.items()}
for k,v in P.items():
    print("%-6s MAE431 %.2f  mean %.1f" % (k, np.abs(v-ye).mean(), v.mean()))
import itertools
keys = list(P)
for r in (2,3):
    for combo in itertools.combinations(keys, r):
        avg = sum(P[k] for k in combo)/r
        print("+".join(combo), "MAE431 %.2f" % np.abs(avg-ye).mean())
# median-of-3
print("median3 MAE431 %.2f" % np.abs(pd.concat(P.values(),axis=1).median(axis=1)-ye).mean())
