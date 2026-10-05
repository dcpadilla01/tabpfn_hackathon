import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)
BASE = dict(n_estimators=900, learning_rate=0.03, max_depth=6, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", n_jobs=4)

DAYS = [375, 403, 431]
def preds(obj, q, train_max):
    tr = df[df.snapshot_day <= train_max]
    m = xgb.XGBRegressor(objective=obj, quantile_alpha=q, **BASE)
    m.fit(tr[FE], tr.future_spend_4w)
    return {d: np.clip(m.predict(df[df.snapshot_day==d][FE]),0,None) for d in DAYS}

t0=time.time()
P_q50 = {}; P_q60 = {}; P_hub = {}
for d in DAYS:
    P_q50.update(preds("reg:quantileerror", 0.5, d-28))
    P_q60.update(preds("reg:quantileerror", 0.6, d-28))
    P_hub.update(preds("reg:pseudohubererror", 0.5, d-28))
print(f"fit time {time.time()-t0:.0f}s")

act = {d: df[df.snapshot_day==d].future_spend_4w.values for d in DAYS}
def ev(getter, label):
    maes = [mean_absolute_error(act[d], getter(d)) for d in DAYS]
    print(f"{label:28s} MAEs {[f'{m:.1f}' for m in maes]} avg {np.mean(maes):.3f}")
ev(lambda d: P_q50[d], "q50")
for s in [1.05, 1.10, 1.15, 1.20]:
    ev(lambda d, s=s: P_q50[d]*s, f"q50 x{s}")
ev(lambda d: P_q60[d], "q60")
for w in [0.3, 0.5, 0.7]:
    ev(lambda d, w=w: w*P_q50[d] + (1-w)*P_hub[d], f"blend q50+hub w={w}")
ev(lambda d: P_hub[d], "huber alone")
