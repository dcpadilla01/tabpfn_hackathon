import pandas as pd, numpy as np, xgboost as xgb, time
from agent_api import load_saved, train_targets

F = load_saved("allF.parquet")
T = train_targets()
feats = [c for c in F.columns if c not in ("household_key","snapshot_day")]
print("allF", F.shape, "n_feats", len(feats))
print(F[feats].dtypes.value_counts().to_dict())
tr = F.merge(T, on=["household_key","snapshot_day"], how="inner")
print("train rows", len(tr), "snaps", sorted(tr.snapshot_day.unique()))
y = tr.future_spend_4w
print("zero share %.3f" % (y==0).mean())
print(y.describe([.25,.5,.75,.9,.95,.99]).round(1))

VAL = F[F.snapshot_day.isin([459,487,515,543])][["household_key","snapshot_day"]]
print("val rows", len(VAL))

def wts(days, ref, half=140.0):
    return 0.5 ** ((np.asarray(ref) - np.asarray(days)) / half)

def fit(df, kind, rounds=1200, lr=0.03, seed=1, ref=None):
    ref = ref if ref is not None else df.snapshot_day.max()
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
    elif kind == "raw":
        lab = df.future_spend_4w.values
        params = dict(objective="reg:squarederror", eta=lr, max_depth=6, min_child_weight=5,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    dtr = xgb.DMatrix(df[feats], label=lab, weight=w)
    bst = xgb.train(params, dtr, num_boost_round=rounds)
    return bst

def pred(bst, df, kind):
    p = bst.predict(xgb.DMatrix(df[feats]))
    if kind == "log": p = np.expm1(p)
    return np.clip(p, 0, None)

t0=time.time()
TRN = tr[tr.snapshot_day <= 375].copy()
HOLD = tr[tr.snapshot_day.isin([403,431])].copy()
print("tune-train rows", len(TRN), "hold rows", len(HOLD))

models = {}
for kind in ["q","log","raw"]:
    t1=time.time()
    b = fit(TRN, kind)
    models[kind] = b
    ph = pred(b, HOLD, kind)
    mae = np.abs(ph - HOLD.future_spend_4w.values).mean()
    print(kind, "hold MAE %.3f  (%.0fs)" % (mae, time.time()-t1),
          "pred mean %.1f vs y mean %.1f" % (ph.mean(), HOLD.future_spend_4w.mean()))
print("total %.0fs" % (time.time()-t0))

P = {k: pred(b, HOLD, k) for k,b in models.items()}
yv = HOLD.future_spend_4w.values
# calibration factors for log/raw on holdout
for k in ["log","raw"]:
    print(k, "mean ratio y/p %.3f  median ratio %.3f" % (yv.mean()/P[k].mean(), np.median(yv)/np.median(P[k])))

print("\nblend grid (w on log model), pooled 403+431:")
best=(1e9,None)
for w in np.arange(0,1.01,0.1):
    for cal in [1.0]:
        pl = P["log"]*cal
        m = np.abs(((1-w)*P["q"] + w*pl) - yv).mean()
        if m<best[0]: best=(m,(w,cal))
        print("w=%.1f MAE %.3f" % (w,m), end="  |  ")
print("\nbest", best)
# per-snapshot best w
for d in [403,431]:
    m = HOLD.snapshot_day.values==d
    ws = [np.abs(((1-w)*P["q"][m] + w*P["log"][m]) - yv[m]).mean() for w in np.arange(0,1.01,0.1)]
    print("day", d, "MAE by w:", np.round(ws,2))
