import pandas as pd, numpy as np, agent_api, xgboost as xgb, time
from sklearn.metrics import mean_absolute_error

feats = agent_api.load_saved("feats_v3.parquet")
tt = agent_api.train_targets()
df = feats.merge(tt, on=["household_key","snapshot_day"], how="left")
FE = [c for c in feats.columns if c not in ("index","household_key","snapshot_day","future_spend_4w")]
df[FE] = df[FE].astype(float)

print("zero frac in train targets:", (df.future_spend_4w==0).mean())

BASE = dict(learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=4)
def fit_eval(cfg, train_max, eval_day, obj="reg:quantileerror", q=0.5):
    tr = df[df.snapshot_day <= train_max]; pv = df[df.snapshot_day == eval_day]
    p = dict(BASE); p.update(cfg)
    m = xgb.XGBRegressor(objective=obj, quantile_alpha=q, **p)
    m.fit(tr[FE], tr.future_spend_4w)
    pr = np.clip(m.predict(pv[FE]), 0, None)
    return mean_absolute_error(pv.future_spend_4w, pr), pr

t0=time.time()
# E005 config as reference on 431
mae0, pr0 = fit_eval(dict(n_estimators=900, max_depth=6, min_child_weight=5), 403, 431)
print(f"E005 cfg on 431: {mae0:.3f}  min pred {pr0.min():.2f}  ({time.time()-t0:.0f}s)")

# bias check per day (rolling origin)
for d in [347, 375, 403, 431]:
    mae, pr = fit_eval(dict(n_estimators=900, max_depth=6, min_child_weight=5), d-28, d)
    act = df[df.snapshot_day==d].future_spend_4w
    print(f"day {d}: MAE {mae:.2f}  mean pred {pr.mean():.1f} vs mean act {act.mean():.1f}  median pred {np.median(pr):.1f} vs {act.median():.1f}")
