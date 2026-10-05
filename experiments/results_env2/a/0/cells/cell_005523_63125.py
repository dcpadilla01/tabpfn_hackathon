
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

f4 = agent_api.load_saved("feats_v4.parquet")
tt = agent_api.train_targets()
df = f4.merge(tt, on=["household_key","snapshot_day"])
y = df.future_spend_4w.values.astype(float)
print("mean/median y by snapshot day:")
g = df.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.round(1))

feats = [c for c in df.columns if c not in ["household_key","snapshot_day","future_spend_4w"]]
cat_cols = [c for c in feats if df[c].dtype == object]
X = pd.get_dummies(df[feats], columns=cat_cols, dummy_na=True)
tr = df.snapshot_day <= 403; va = df.snapshot_day == 431
ytr, yva = y[tr.values], y[va.values]
Xtr, Xva = X[tr.values], X[va.values]

def fit(obj="reg:quantileerror", alpha=0.5, depth=5, mcw=40, seed=0, n=400, lr=0.08, Xtr_=Xtr, ytr_=ytr):
    m = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, min_child_weight=mcw,
                     subsample=0.8, colsample_bytree=0.8, objective=obj,
                     quantile_alpha=alpha if "quantile" in obj else None,
                     random_state=seed, n_jobs=4, tree_method="hist")
    m.fit(Xtr_, ytr_)
    return np.clip(m.predict(Xva), 0, None)

p = fit()
# error by y decile
qs = np.quantile(yva, np.linspace(0,1,11))
print("\nday431 error by y-decile (base quantile model):")
for i in range(10):
    lo, hi = qs[i], qs[i+1]
    msk = (yva >= lo) & (yva <= hi) if i==9 else (yva >= lo) & (yva < hi)
    print(f"  dec {i}: y [{lo:6.1f},{hi:6.1f}] n={msk.sum():4d} MAE={np.abs(p[msk]-yva[msk]).mean():7.2f} bias={(p[msk]-yva[msk]).mean():8.2f}")

# bag: quantile + absoluteerror objective, seeds
def bag(configs, seeds=(0,1)):
    ps = []
    for c in configs:
        for s in seeds:
            ps.append(fit(**c, seed=s))
    return np.mean(ps, axis=0)

cfgs = [dict(depth=d, mcw=w) for d in (4,5,6) for w in (20,40,60)][:8]
pb2 = bag(cfgs, seeds=(0,1)); pb4 = bag(cfgs, seeds=(0,1,2,3))
print("\nquantile bag 2 seeds MAE", round(np.abs(pb2-yva).mean(),3))
print("quantile bag 4 seeds MAE", round(np.abs(pb4-yva).mean(),3))
cfgs_l1 = cfgs[:4] + [dict(obj="reg:absoluteerror", depth=d, mcw=w) for d,w in [(4,20),(5,40),(6,40),(4,60)]]
pb_l1 = bag(cfgs_l1, seeds=(0,1))
print("quantile+L1 mixed bag MAE", round(np.abs(pb_l1-yva).mean(),3))
