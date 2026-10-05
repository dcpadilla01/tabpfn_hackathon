import pandas as pd, numpy as np, xgboost as xgb, time
from agent_api import load_saved

F = load_saved("allF.parquet")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
print("n_feats", len(feats))
tr = F[F.future_spend_4w.notna()].copy()   # train snapshots only
print("train rows", len(tr), sorted(tr.snapshot_day.unique()))

def wts(days, ref, half=140.0):
    return 0.5 ** ((np.asarray(ref) - np.asarray(days)) / half)

def fit(df, kind, rounds=1200, lr=0.03, seed=1):
    ref = df.snapshot_day.max()
    w = wts(df.snapshot_day, ref)
    if kind == "q":
        lab = df.future_spend_4w.values
        params = dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=lr,
                      max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", seed=seed)
    elif kind == "log":
        lab = np.log1p(df.future_spend_4w.values)
        params = dict(objective="reg:squarederror", eta=lr, max_depth=6, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    else:
        lab = df.future_spend_4w.values
        params = dict(objective="reg:squarederror", eta=lr, max_depth=6, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    dtr = xgb.DMatrix(df[feats], label=lab, weight=w)
    return xgb.train(params, dtr, num_boost_round=rounds)

def pred(bst, df, kind):
    p = bst.predict(xgb.DMatrix(df[feats]))
    if kind == "log": p = np.expm1(p)
    return np.clip(p, 0, None)

t0=time.time()
TRN = tr[tr.snapshot_day <= 375].copy()
HOLD = tr[tr.snapshot_day.isin([403,431])].copy()
yv = HOLD.future_spend_4w.values
P = {}
for kind in ["q","log","raw"]:
    t1=time.time(); b = fit(TRN, kind); P[kind] = pred(b, HOLD, kind)
    print(kind, "hold MAE %.3f (%.0fs) pred mean %.1f y mean %.1f" %
          (np.abs(P[kind]-yv).mean(), time.time()-t1, P[kind].mean(), yv.mean()))
print("total %.0fs" % (time.time()-t0))
for k in ["log","raw"]:
    print(k, "mean ratio y/p %.3f med ratio %.3f" % (yv.mean()/P[k].mean(), np.median(yv)/np.median(P[k])))
print("\nblend grid:")
for w in np.arange(0,1.01,0.1):
    print("w=%.1f MAE %.3f" % (w, np.abs((1-w)*P["q"] + w*P["log"] - yv).mean()))
for d in [403,431]:
    m = HOLD.snapshot_day.values==d
    print("day", d, np.round([np.abs((1-w)*P["q"][m]+w*P["log"][m]-yv[m]).mean() for w in np.arange(0,1.01,0.1)],2))
