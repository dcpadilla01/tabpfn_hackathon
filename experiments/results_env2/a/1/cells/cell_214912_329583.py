
import pandas as pd, numpy as np, xgboost as xgb

f = agent_api.load_saved("e002_features.parquet")
tt = agent_api.train_targets().set_index(["household_key","snapshot_day"])
f = f.join(tt.future_spend_4w.rename("y"), on=["household_key","snapshot_day"])
train = f[f.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403])].dropna(subset=["y"])
hold = f[f.snapshot_day==431].dropna(subset=["y"])
print(len(train), len(hold))

drop = ["index","household_key","snapshot_day","y"]
Xcols = [c for c in f.columns if c not in drop]
Xtr, ytr = train[Xcols], train.y
Xho, yho = hold[Xcols], hold.y
print("naive s84/3 holdout MAE:", round((hold.spend_84/3 - yho).abs().mean(),3))
print("naive s28 holdout MAE:", round((hold.spend_28 - yho).abs().mean(),3))

def fit_eval(params, Xtr=Xtr, ytr=ytr, Xho=Xho, yho=yho, num=800):
    m = xgb.XGBRegressor(n_estimators=num, tree_method="hist", n_jobs=8, **params)
    m.fit(Xtr, ytr)
    p = m.predict(Xho)
    return round(np.abs(p-yho).mean(),3), m, p

for obj in ["reg:squarederror","reg:absoluteerror","reg:pseudohubererror"]:
    for lr, depth in [(0.05,6),(0.03,5)]:
        mae,_,_ = fit_eval(dict(objective=obj, learning_rate=lr, max_depth=depth, subsample=0.8, colsample_bytree=0.7, reg_lambda=5))
        print(obj, lr, depth, "MAE:", mae)
