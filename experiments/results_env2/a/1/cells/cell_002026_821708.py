
import pandas as pd, numpy as np, time
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

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
Xtr, ytr = mat(tr_days)
w = pd.Series(0.5 ** ((431 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
X431, y431 = mat([431])

t0=time.time()
mq = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.03,
                  n_estimators=2400, max_depth=6, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
mq.fit(Xtr, ytr, sample_weight=w)
p = pd.Series(mq.predict(X431), index=X431.index).clip(lower=0)
print("fit q %.0fs; raw median MAE 431: %.3f" % (time.time()-t0, np.abs(p-y431).mean()))

mz = XGBRegressor(objective="reg:logistic", learning_rate=0.03, n_estimators=1200,
                  max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=5.0, tree_method="hist", n_jobs=8)
mz.fit(Xtr, (ytr>0).astype(int), sample_weight=w)
q = pd.Series(mz.predict(X431), index=X431.index)
chk = pd.DataFrame({"q":q, "pos":(y431>0).astype(int)})
print("mean q by actual pos:", chk.groupby("pos").q.mean().round(3).to_dict())
print("q deciles vs pos rate:", chk.assign(b=pd.qcut(q,10,duplicates="drop")).groupby("b",observed=True).pos.mean().round(2).to_dict())

one = pd.Series(1.0, index=q.index)
for name, s in [("p*(1-q)", p*(1-q)), ("p*(1-q)^.5", p*(1-q)**.5), ("p*(1-q)^2", p*(1-q)**2),
                ("hard0", p.where(q<0.5, 0.0)), ("hard.35", p.where(q<0.35, 0.0)),
                ("blend.5", 0.5*p + 0.5*p*(1-q))]:
    print("%-10s MAE 431: %.3f" % (name, np.abs(s-y431).mean()))
