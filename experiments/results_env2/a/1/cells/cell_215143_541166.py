
import pandas as pd, numpy as np, xgboost as xgb
f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
print(f.groupby("snapshot_day").y.agg(["mean","median"]).round(1).T)
Xcols = [c for c in f.columns if c not in ["index","household_key","snapshot_day","y"]]
tr = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
ho = f[f.snapshot_day==431].dropna(subset=["y"])
Xtr,ytr,Xho,yho = tr[Xcols], tr.y, ho[Xcols], ho.y
def mae(p): return round(float(np.abs(p-yho).mean()),3)
def qmodel(alpha, lr=0.03, depth=5, mcw=5, num=1200):
    m = xgb.XGBRegressor(n_estimators=num, objective="reg:quantileerror", quantile_alpha=alpha,
                         learning_rate=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8,
                         colsample_bytree=0.7, reg_lambda=5, tree_method="hist", n_jobs=8)
    m.fit(Xtr, ytr); return m
preds = {}
for a in [0.45,0.5,0.55,0.6]:
    m = qmodel(a); p = m.predict(Xho); preds[a]=p; print("q",a, mae(p))
# ensemble of quantile + skHGB
from sklearn.ensemble import HistGradientBoostingRegressor
hg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=600, learning_rate=0.05, min_samples_leaf=40,
                                   l2_regularization=1.0, random_state=0)
hg.fit(Xtr, ytr); ph = hg.predict(Xho); print("skHGB:", mae(ph))
for w in [0.3,0.5,0.7]:
    print("ens q0.5+skHGB w=",w, mae(w*preds[0.5]+(1-w)*ph))
ens = np.mean([preds[a] for a in preds]); print("ens all q:", mae(ens))
# bias check by snapshot for q0.5
m50 = qmodel(0.5)
val_all = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431])].dropna(subset=["y"])
pv = m50.predict(val_all[Xcols])
r = pd.DataFrame({"d":val_all.snapshot_day,"p":pv,"y":val_all.y})
print(r.groupby("d").apply(lambda g: pd.Series({"bias":(g.p-g.y).mean(),"mae":(g.p-g.y).abs().mean()}), include_groups=False).round(2))
