import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr = F[F.future_spend_4w.notna()].copy()
TRN = tr[tr.snapshot_day <= 375]
HOLD = tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_pred(df_tr, df_ho, kind="q", rounds=1200, lr=0.03, half=140.0, seed=1, depth=6, mcw=5):
    w = 0.5 ** ((df_tr.snapshot_day.values - df_tr.snapshot_day.max()) / half)
    if kind=="q": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:quantileerror", quantile_alpha=0.5)
    elif kind=="qlog": lab, obj = np.log1p(df_tr.future_spend_4w.values), dict(objective="reg:quantileerror", quantile_alpha=0.5)
    elif kind=="huber": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:pseudohubererror", huber_slope=20.0)
    elif kind=="l2": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:squarederror")
    p = dict(eta=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", seed=seed); p.update(obj)
    b = xgb.train(p, xgb.DMatrix(df_tr[feats], label=lab, weight=w), num_boost_round=rounds)
    pr = b.predict(xgb.DMatrix(df_ho[feats]))
    if kind=="qlog": pr = np.expm1(pr)
    return np.clip(pr, 0, None)

t0=time.time(); res={}
cfgs = [
  ("q h140 d6",      dict(kind="q", half=140)),
  ("q h100",         dict(kind="q", half=100)),
  ("q h200",         dict(kind="q", half=200)),
  ("q h277",         dict(kind="q", half=277)),
  ("qlog h140",      dict(kind="qlog", half=140)),
  ("huber h140",     dict(kind="huber", half=140)),
  ("q h140 d8",      dict(kind="q", half=140, depth=8)),
  ("q h140 r2400",   dict(kind="q", half=140, rounds=2400)),
]
for name, kw in cfgs:
    p = fit_pred(TRN, HOLD, **kw)
    res[name] = p
    print("%-14s MAE %.3f  mean %.1f" % (name, np.abs(p-yv).mean(), p.mean()))
print("%.0fs" % (time.time()-t0))

print("\ncalibration scale on q h140:")
pq = res["q h140 d6"]
for c in [0.95,1.0,1.05,1.1,1.15]:
    print("c=%.2f MAE %.3f" % (c, np.abs(c*pq-yv).mean()))
print("\nblend q + qlog:")
pl = res["qlog h140"]
for w in np.arange(0,1.01,0.25):
    print("w_qlog=%.2f MAE %.3f" % (w, np.abs((1-w)*pq + w*pl - yv).mean()))
print("\nblend q + huber:")
ph = res["huber h140"]
for w in np.arange(0,1.01,0.25):
    print("w_hub=%.2f MAE %.3f" % (w, np.abs((1-w)*pq + w*ph - yv).mean()))
