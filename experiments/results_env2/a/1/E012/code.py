import pandas as pd, numpy as np
from agent_api import load_saved

names = ["e001_preds","e002_preds","e003_preds","e004_preds","e005_preds","e007_preds","e008_preds","e009_preds","e010_preds","e011_preds"]
P = {}
for n in names:
    df = load_saved(n + ".parquet")
    P[n] = df
    print(n, df.shape, list(df.columns))

base = P["e005_preds"][["household_key","snapshot_day"]].copy()
print("base rows", len(base), "unique hh", base.household_key.nunique(), "days", sorted(base.snapshot_day.unique()))

M = base.copy()
for n, df in P.items():
    d = df[["household_key","snapshot_day","prediction"]].rename(columns={"prediction": n})
    M = M.merge(d, on=["household_key","snapshot_day"], how="left")
cols = names
print("NaNs:", M[cols].isna().sum().to_dict())
print(M[cols].describe().loc[["mean","std","min","max"]].round(2))
print(M[cols].corr().round(3))

for n in ["repro_e5","e005_newfeats","e004_new","allF","lagfeats","lagfeats2","f_weekly","e004_features","e002_features"]:
    try:
        df = load_saved(n + ".parquet")
        print("---", n, df.shape, list(df.columns)[:30])
    except Exception as e:
        print(n, "ERR", type(e).__name__, str(e)[:80])


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved, train_targets
T = train_targets()
print(T.columns.tolist(), T.shape)
print(T.head(3))
F = load_saved("allF.parquet")
tr = F.merge(T, on=["household_key","snapshot_day"], how="inner")
print(tr.columns[-5:].tolist(), len(tr))


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np
from agent_api import load_saved
F = load_saved("allF.parquet")
print([c for c in F.columns])


# ---- cell ----
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


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet")
W = load_saved("f_weekly.parquet")
print(W.columns.tolist())
# columns already in allF or tested before
skip = {"household_key","snapshot_day","lag28_56","lag56_84","lag84_112","sp28_yag","sp84_yag"}
newcols = [c for c in W.columns if c not in skip]
print("new cols:", newcols)
F2 = F.merge(W[["household_key","snapshot_day"]+newcols], on=["household_key","snapshot_day"], how="left")
print(F2.shape, "NaN new:", F2[newcols].isna().sum().sum())

tr = F2[F2.future_spend_4w.notna()].copy()
TRN = tr[tr.snapshot_day <= 375]
HOLD = tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_pred(cols, df_tr, df_ho, kind="q", rounds=1200, lr=0.03, half=140.0, seed=1, depth=6, mcw=5):
    w = 0.5 ** ((df_tr.snapshot_day.values - df_tr.snapshot_day.max()) / half)
    if kind=="q": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:quantileerror", quantile_alpha=0.5)
    elif kind=="l1": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:absoluteerror")
    elif kind=="sqrt": lab, obj = np.sqrt(df_tr.future_spend_4w.values), dict(objective="reg:squarederror")
    p = dict(eta=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", seed=seed); p.update(obj)
    b = xgb.train(p, xgb.DMatrix(df_tr[cols], label=lab, weight=w), num_boost_round=rounds)
    pr = b.predict(xgb.DMatrix(df_ho[cols]))
    if kind=="sqrt": pr = np.square(pr)
    return np.clip(pr, 0, None)

base_cols = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
new_cols = base_cols + newcols
t0=time.time()
for name, cols, kw in [
    ("base q",        base_cols, dict(kind="q")),
    ("base+weekly q", new_cols,  dict(kind="q")),
    ("base l1",       base_cols, dict(kind="l1")),
    ("base+weekly l1",new_cols,  dict(kind="l1")),
    ("base sqrt",     base_cols, dict(kind="sqrt")),
]:
    p = fit_pred(cols, TRN, HOLD, **kw)
    print("%-16s MAE %.3f mean %.1f" % (name, np.abs(p-yv).mean(), p.mean()))
print("%.0fs" % (time.time()-t0))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet")
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
tr = F[F.future_spend_4w.notna()].copy()
TRN = tr[tr.snapshot_day <= 375]
HOLD = tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_q(df_tr, alpha=0.5, rounds=1200, lr=0.03, half=140.0, seed=1, depth=6):
    w = 0.5 ** ((df_tr.snapshot_day.values - df_tr.snapshot_day.max()) / half)
    p = dict(objective="reg:quantileerror", quantile_alpha=alpha, eta=lr, max_depth=depth,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(df_tr[feats], label=df_tr.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return np.clip(b.predict(xgb.DMatrix(HOLD[feats])), 0, None)

t0=time.time()
pq = fit_q(TRN)
print("q alpha.5 MAE %.3f" % np.abs(pq-yv).mean())

# 1) alpha sweep (global calibration proxy)
for a in [0.45, 0.55, 0.60]:
    p = fit_q(TRN, alpha=a)
    print("alpha %.2f MAE %.3f mean %.1f" % (a, np.abs(p-yv).mean(), p.mean()))

# 2) residual boosting: fit L2 on residual of q model, using train preds
w = 0.5 ** ((TRN.snapshot_day.values - TRN.snapshot_day.max()) / 140)
p = dict(objective="reg:squarederror", eta=0.03, max_depth=5, min_child_weight=10,
         subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=2)
bq = xgb.train(dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=1),
               xgb.DMatrix(TRN[feats], label=TRN.future_spend_4w.values, weight=w), num_boost_round=1200)
ptr = bq.predict(xgb.DMatrix(TRN[feats]))
resid = TRN.future_spend_4w.values - ptr
for nr in [300, 600]:
    br = xgb.train(p, xgb.DMatrix(TRN[feats], label=resid, weight=w), num_boost_round=nr)
    pr = bq.predict(xgb.DMatrix(HOLD[feats])) + br.predict(xgb.DMatrix(HOLD[feats]))
    pr = np.clip(pr, 0, None)
    print("resid boost %d MAE %.3f mean %.1f" % (nr, np.abs(pr-yv).mean(), pr.mean()))

# 3) level-binned recalibration (fit on holdout itself = upper bound of gain)
bins = np.quantile(pq, np.linspace(0,1,11))
idx = np.clip(np.digitize(pq, bins[1:-1]), 0, 9)
corr = np.array([np.median(yv[idx==i] - pq[idx==i]) if (idx==i).sum()>5 else 0 for i in range(10)])
pc = pq + corr[idx]
print("binned recal (in-sample) MAE %.3f  (gain upper bound)" % np.abs(pc-yv).mean())
print("bin medians resid:", np.round(corr,1))
# 4) isotonic y~p
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip").fit(pq, yv)
pi = iso.predict(pq)
print("isotonic (in-sample) MAE %.3f" % np.abs(pi-yv).mean())
print("%.0fs" % (time.time()-t0))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, load_saved, save_table

def camp_feats(view, snapshot_day):
    hh = view.households
    s = int(snapshot_day)
    camp = view.table("campaigns")          # start_day <= s
    ct = view.table("campaign_targets")     # households targeted
    m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
    m = m[m.household_key.isin(set(hh))]
    g = m.groupby("household_key")
    out = pd.DataFrame(index=pd.Index(sorted(set(hh)), name="household_key"))
    n_active   = g.apply(lambda d: ((d.end_day >= s)).sum(), include_groups=False)
    n_start28  = g.apply(lambda d: ((d.start_day >= s-27) & (d.start_day <= s)).sum(), include_groups=False)
    n_endfut   = g.apply(lambda d: ((d.end_day > s) & (d.end_day <= s+28)).sum(), include_groups=False)
    camp_days  = g.apply(lambda d: np.clip(d.end_day, s+1, None).clip(upper=s+28) - np.clip(d.start_day, s+1, s+28)).clip(lower=0).groupby(level=0).sum()
    last_start = g.start_day.max()
    out["cmp_active"]   = n_active.reindex(out.index).fillna(0)
    out["cmp_start28"]  = n_start28.reindex(out.index).fillna(0)
    out["cmp_endfut"]   = n_endfut.reindex(out.index).fillna(0)
    out["cmp_fut_days"] = camp_days.reindex(out.index).fillna(0)
    out["cmp_laststart"] = (s - last_start).reindex(out.index).fillna(9999)
    ta = m[m.description=="TypeA"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    tb = m[m.description=="TypeB"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    tc = m[m.description=="TypeC"].groupby("household_key").apply(lambda d: (d.end_day>=s).sum(), include_groups=False)
    out["cmp_act_A"] = ta.reindex(out.index).fillna(0)
    out["cmp_act_B"] = tb.reindex(out.index).fillna(0)
    out["cmp_act_C"] = tc.reindex(out.index).fillna(0)
    return out

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, CF.columns.tolist(), "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day")[["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days"]].mean().round(2))
save_table(CF.reset_index(), "camp_feats")


# ---- cell ----
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, save_table

def camp_feats(view, snapshot_day):
    hh = set(view.households)
    s = int(snapshot_day)
    camp = view.table("campaigns")
    ct = view.table("campaign_targets")
    m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
    m["start_day"] = pd.to_numeric(m["start_day"]); m["end_day"] = pd.to_numeric(m["end_day"])
    m = m[m.household_key.isin(hh)]
    idx = pd.Index(sorted(hh), name="household_key")
    g = m.groupby("household_key")
    out = pd.DataFrame(index=idx)
    out["cmp_active"]   = g.apply(lambda d: float((d.end_day >= s).sum()), include_groups=False).reindex(idx).fillna(0)
    out["cmp_start28"]  = g.apply(lambda d: float(((d.start_day >= s-27) & (d.start_day <= s)).sum()), include_groups=False).reindex(idx).fillna(0)
    out["cmp_endfut"]   = g.apply(lambda d: float(((d.end_day > s) & (d.end_day <= s+28)).sum()), include_groups=False).reindex(idx).fillna(0)
    def fut_days(d):
        a = np.maximum(pd.to_numeric(d.start_day).values, s+1)
        b = np.minimum(pd.to_numeric(d.end_day).values, s+28)
        return float(np.clip(b - a + 1, 0, None).sum())
    out["cmp_fut_days"] = g.apply(fut_days, include_groups=False).reindex(idx).fillna(0)
    out["cmp_laststart"] = (s - g.start_day.max()).reindex(idx).fillna(9999).astype(float)
    for t in ["TypeA","TypeB","TypeC"]:
        mt = m[m.description==t]
        out["cmp_act_"+t[-1]] = mt.groupby("household_key").apply(lambda d: float((d.end_day>=s).sum()), include_groups=False).reindex(idx).fillna(0)
    return out

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day")[["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days","cmp_act_A","cmp_act_B","cmp_act_C"]].mean().round(2))
save_table(CF.reset_index(), "camp_feats")


# ---- cell ----
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, save_table

def num(s):
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(dtype=float)

def camp_feats(view, snapshot_day):
    hh = sorted(set(int(x) for x in view.households))
    s = int(snapshot_day)
    camp = view.table("campaigns")
    ct = view.table("campaign_targets")
    m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
    m["start_day"] = num(m["start_day"]); m["end_day"] = num(m["end_day"])
    m["household_key"] = m["household_key"].astype(int)
    m["description"] = m["description"].astype(str)
    m = m[m.household_key.isin(hh)]
    idx = pd.Index(hh, name="household_key")
    g = m.groupby("household_key")
    out = pd.DataFrame(index=idx)
    def col(series_or_arr):
        return np.asarray(series_or_arr, dtype=float)
    out["cmp_active"]  = col(g.apply(lambda d: float((d.end_day >= s).sum()), include_groups=False).reindex(idx).fillna(0))
    out["cmp_start28"] = col(g.apply(lambda d: float(((d.start_day >= s-27) & (d.start_day <= s)).sum()), include_groups=False).reindex(idx).fillna(0))
    out["cmp_endfut"]  = col(g.apply(lambda d: float(((d.end_day > s) & (d.end_day <= s+28)).sum()), include_groups=False).reindex(idx).fillna(0))
    def fut_days(d):
        a = np.maximum(d.start_day.values, s+1); b = np.minimum(d.end_day.values, s+28)
        return float(np.clip(b - a + 1, 0, None).sum())
    out["cmp_fut_days"] = col(g.apply(fut_days, include_groups=False).reindex(idx).fillna(0))
    out["cmp_laststart"] = col((s - g.start_day.max()).reindex(idx).fillna(9999))
    for t in ["TypeA","TypeB","TypeC"]:
        mt = m[m.description == t]
        out["cmp_act_" + t[-1]] = col(mt.groupby("household_key").apply(lambda d: float((d.end_day >= s).sum()), include_groups=False).reindex(idx).fillna(0))
    return out

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, "%.0fs" % (time.time()-t0))
print(CF.dtypes.unique())
print(CF.groupby("snapshot_day")[["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days","cmp_act_A","cmp_act_B","cmp_act_C"]].mean().round(2))
save_table(CF.reset_index(), "camp_feats")


# ---- cell ----
import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from agent_api import build_features, save_table

def camp_feats(view, snapshot_day):
    hh = sorted(set(int(x) for x in view.households))
    s = int(snapshot_day)
    idx = pd.Index(hh, name="household_key")
    try:
        camp = view.table("campaigns"); ct = view.table("campaign_targets")
        m = ct.merge(camp[["campaign","start_day","end_day"]], on="campaign", how="left")
        st = pd.to_numeric(m["start_day"], errors="coerce").astype(float).values
        en = pd.to_numeric(m["end_day"], errors="coerce").astype(float).values
        m = m.assign(st=st, en=en)
        m["hh"] = m["household_key"].astype(int).values
        m = m[m["hh"].isin(set(hh))]
        m["active"]   = (m.en >= s).astype(float)
        m["start28"]  = ((m.st >= s-27) & (m.st <= s)).astype(float)
        m["endfut"]   = ((m.en > s) & (m.en <= s+28)).astype(float)
        m["futdays"]  = np.clip(np.minimum(m.en, s+28) - np.maximum(m.st, s+1) + 1, 0, None)
        m["desc"] = m["description"].astype(str)
        g = m.groupby("hh")
        out = pd.DataFrame(index=idx)
        for c in ["active","start28","endfut","futdays"]:
            out["cmp_"+c] = g[c].sum().reindex(idx).fillna(0.0).astype(float)
        out["cmp_laststart"] = (s - g["st"].max()).reindex(idx).fillna(9999.0).astype(float)
        for t in ["TypeA","TypeB","TypeC"]:
            mt = m[m["desc"]==t]
            out["cmp_act_"+t[-1]] = mt.groupby("hh")["active"].sum().reindex(idx).fillna(0.0).astype(float)
        if s >= 95:
            print("camp feats ok", out.shape, out.mean().round(2).to_dict())
        return out
    except Exception as e:
        print("camp fallback:", type(e).__name__, str(e)[:100])
        return pd.DataFrame(0.0, index=idx, columns=["cmp_active","cmp_start28","cmp_endfut","cmp_futdays","cmp_laststart","cmp_act_A","cmp_act_B","cmp_act_C"])

t0=time.time()
CF = build_features(camp_feats)
print(CF.shape, "%.0fs" % (time.time()-t0))
print(CF.groupby("snapshot_day").mean().round(2))
save_table(CF.reset_index(), "camp_feats")


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet"); C = load_saved("camp_feats.parquet")
F2 = F.merge(C, on=["household_key","snapshot_day"], how="left").fillna({"cmp_laststart":9999.0})
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
feats2 = feats + ["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days","cmp_laststart","cmp_act_A","cmp_act_B","cmp_act_C"]
tr = F2[F2.future_spend_4w.notna()].copy()
TRN, HOLD = tr[tr.snapshot_day<=375], tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_pred(cols, df_tr, df_ho, half=140.0, rounds=1200, seed=1):
    w = 0.5**((df_tr.snapshot_day.values-df_tr.snapshot_day.max())/half)
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(df_tr[cols], label=df_tr.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return np.clip(b.predict(xgb.DMatrix(df_ho[cols])),0,None)

t0=time.time()
p1 = fit_pred(feats, TRN, HOLD)
p2 = fit_pred(feats2, TRN, HOLD)
print("base MAE %.3f | +camp MAE %.3f  (%.0fs)" % (np.abs(p1-yv).mean(), np.abs(p2-yv).mean(), time.time()-t0))


# ---- cell ----
import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved, save_table

F = load_saved("allF.parquet"); C = load_saved("camp_feats.parquet")
camp_cols = [c for c in C.columns if c not in ("household_key","snapshot_day")]
print("camp cols:", camp_cols)
F2 = F.merge(C, on=["household_key","snapshot_day"], how="left")
for c in camp_cols:
    F2[c] = pd.to_numeric(F2[c], errors="coerce").astype(float)
fill = {c: 0.0 for c in camp_cols}; fill["cmp_laststart"] = 9999.0
F2 = F2.fillna(fill)
feats = [c for c in F2.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
print("n_feats", len(feats))

tr = F2[F2.future_spend_4w.notna()]
TRN = tr[tr.snapshot_day <= 375].copy()
VAL = F2[F2.snapshot_day.isin([459,487,515,543])].copy()
print("TRN", TRN.shape, "VAL", VAL.shape, sorted(VAL.snapshot_day.unique()))

def fit_q(seed, rounds=2400, half=140.0):
    w = 0.5 ** ((TRN.snapshot_day.values - 375) / half)
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(TRN[feats], label=TRN.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return b.predict(xgb.DMatrix(VAL[feats]))

t0 = time.time()
ps = [fit_q(s) for s in (1, 2, 3, 4)]
pred = np.clip(np.mean(ps, axis=0), 0, None)
print("pred stats: mean %.1f std %.1f min %.1f max %.1f  (%.0fs)" % (pred.mean(), pred.std(), pred.min(), pred.max(), time.time()-t0))
assert np.isfinite(pred).all() and len(pred) == len(VAL)
out = pd.DataFrame({"household_key": VAL.household_key.values.astype(int),
                    "snapshot_day": VAL.snapshot_day.values.astype(int),
                    "prediction": pred.astype(float)})
path = save_table(out, "e012_preds")
print("saved", path, out.shape)
