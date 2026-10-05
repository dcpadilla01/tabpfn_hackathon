
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
drop = ["household_key","snapshot_day","future_spend_4w"]
feats = [c for c in df.columns if c not in drop]
# one-hot the categoricals
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
y = df.future_spend_4w.values.astype(float)

tr = df.snapshot_day <= 403
va = df.snapshot_day == 431
Xtr, ytr, Xva, yva = X[tr.values], y[tr.values], X[va.values], y[va.values]
print("inner tr/va", Xtr.shape, Xva.shape)
print("e013-style baseline check: exp4w_blend MAE on 431:",
      round(np.abs(df.loc[va,"exp4w_blend"].values - yva).mean(),3))

def run(tag, obj, transform=None, inv=None, blend=0.3, seed=0, depth=5, mcw=40):
    yt = transform(ytr) if transform else ytr
    m = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=depth,
                     min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
                     objective=obj, quantile_alpha=0.5 if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr, yt)
    p = m.predict(Xva)
    p = inv(p) if inv else p
    pb = (1-blend)*p + blend*df.loc[va,"exp4w_blend"].values
    print(f"{tag:28s} MAE {np.abs(p-yva).mean():7.3f}  blended {np.abs(pb-yva).mean():7.3f}")

run("quantile0.5 raw (E011-style)", "reg:quantileerror")
run("sqerr raw", "reg:squarederror")
run("sqerr log1p", "reg:squarederror", transform=np.log1p, inv=np.expm1)
run("quantile0.5 log1p", "reg:quantileerror", transform=np.log1p, inv=np.expm1)
run("pseudohuber raw", "reg:pseudohubererror")
run("sqerr sqrt", "reg:squarederror", transform=np.sqrt, inv=lambda z: np.square(z))
