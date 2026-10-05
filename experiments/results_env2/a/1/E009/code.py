
import pandas as pd, numpy as np

names = ["allF","e001_preds","e002_features","e002_preds","e003_preds","e004_features",
         "e004_new","e004_preds","e005_newfeats","e005_preds","e007_preds","e008_preds",
         "lagfeats","lagfeats2"]
store = {}
for nm in names:
    for arg in (nm, nm+".parquet"):
        try:
            df = load_saved(arg)
            store[nm] = df
            print(nm, df.shape, "|", list(df.columns)[:14])
            break
        except Exception as e:
            last = (arg, type(e).__name__, str(e)[:80])
    else:
        print(nm, "ERR", last)

print()
print("snapshot_days:", snapshot_days())
print("KEYS:", KEYS, "TARGET:", TARGET)

tt = train_targets()
print("train_targets:", tt.shape, tt[TARGET].describe().round(2).to_dict())

# correlation of saved val predictions
p4 = store.get("e004_preds"); p5 = store.get("e005_preds"); p3 = store.get("e003_preds")
if p4 is not None and p5 is not None:
    m = p4.merge(p5, on=["household_key","snapshot_day"], suffixes=("_4","_5"))
    val = m[m.snapshot_day.isin(snapshot_days()["validation"])]
    print("val rows:", len(val))
    print("corr e004 vs e005 (val):", val.prediction_4.corr(val.prediction_5).round(4))
    print("e005 val pred describe:", val.prediction_5.describe().round(2).to_dict())
    if p3 is not None:
        m = m.merge(p3, on=["household_key","snapshot_day"])
        m.rename(columns={"prediction":"pred_3"}, inplace=True)
        v = m[m.snapshot_day.isin(snapshot_days()["validation"])]
        print("corr e003 vs e005 (val):", v.pred_3.corr(v.prediction_5).round(4))


# ---- cell ----

import pandas as pd, numpy as np

tt = train_targets()
tt["zero"] = (tt.future_spend_4w == 0).astype(int)
print("zero fraction overall:", tt.zero.mean().round(4))
print(tt.groupby("snapshot_day").agg(n=("zero","size"), zero_frac=("zero","mean"),
      mean=("future_spend_4w","mean"), med=("future_spend_4w","median")).round(3))

f4 = load_saved("e004_features.parquet")
print("\ne004_features cols:", list(f4.columns))
f4n = load_saved("e004_new.parquet"); f5n = load_saved("e005_newfeats.parquet")
cols4n = [c for c in f4n.columns if c not in ("household_key","snapshot_day")]
cols5n = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in cols4n if c in f4.columns] + [c for c in cols5n if c in f4.columns]
print("dup cols:", dup)
F = f4.merge(f4n.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
dup2 = [c for c in cols5n if c in F.columns]
F = F.merge(f5n.drop(columns=dup2), on=["household_key","snapshot_day"], how="inner")
print("merged F:", F.shape)
print("snapshot_day counts:", F.snapshot_day.value_counts().sort_index().to_dict())

p5 = load_saved("e005_preds.parquet")
v = p5[predcol:=( "prediction" )]
print("\ne005 val preds low tail:", {t: float((p5.prediction < t).mean()) for t in [1,5,10,20,30]})


# ---- cell ----

import pandas as pd, numpy as np, time
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in c5 if c in F.columns]
F = F.merge(f5n.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]
Xtr, ytr = mat(tr_days)
w = pd.Series(0.5 ** ((431 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
X431, y431 = mat([431])

t0=time.time()
mq = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.03,
                  n_estimators=2400, max_depth=6, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
mq.fit(Xtr, ytr, sample_weight=w)
p = pd.Series(mq.predict(X431), index=X431.index).clip(lower=0)
print("fit q %.0fs; raw median MAE 431: %.3f" % (time.time()-t0, np.abs(p-y431).mean()))

mz = XGBRegressor(objective="reg:logistic", learning_rate=0.03, n_estimators=1200,
                  max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
                  reg_lambda=5.0, tree_method="hist", n_jobs=8)
mz.fit(Xtr, (ytr>0).astype(int), sample_weight=w)
q = pd.Series(mz.predict(X431), index=X431.index)
chk = pd.DataFrame({"q":q, "pos":(y431>0).astype(int)})
print("mean q by actual pos:", chk.groupby("pos").q.mean().round(3).to_dict())
print("q deciles vs pos rate:", chk.assign(b=pd.qcut(q,10,duplicates="drop")).groupby("b",observed=True).pos.mean().round(2).to_dict())

one = pd.Series(1.0, index=q.index)
for name, s in [("p*(1-q)", p*(1-q)), ("p*(1-q)^.5", p*(1-q)**.5), ("p*(1-q)^2", p*(1-q)**2),
                ("hard0", p.where(q<0.5, 0.0)), ("hard.35", p.where(q<0.35, 0.0)),
                ("blend.5", 0.5*p + 0.5*p*(1-q))]:
    print("%-10s MAE 431: %.3f" % (name, np.abs(s-y431).mean()))


# ---- cell ----

import pandas as pd, numpy as np
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
dup = [c for c in c5 if c in F.columns]
F = F.merge(f5n.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
mq = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.03,
                  n_estimators=2400, max_depth=6, min_child_weight=10, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
mq.fit(Xtr, ytr, sample_weight=w)

Xc, yc = mat([403]); Xe, ye = mat([431])
pc = pd.Series(mq.predict(Xc), index=Xc.index).clip(lower=0)
pe = pd.Series(mq.predict(Xe), index=Xe.index).clip(lower=0)
print("MAE 403: %.2f  MAE 431: %.2f" % (np.abs(pc-yc).mean(), np.abs(pe-ye).mean()))

# residual bias by decile of p (on 403, the calibration snapshot)
d = pd.DataFrame({"p":pc, "y":yc})
d["b"] = pd.qcut(d.p, 10, duplicates="drop")
print(d.groupby("b", observed=True).agg(n=("y","size"), p=("p","mean"), y=("y","mean"),
      bias=("y","mean")).assign(bias=lambda x: x.y - x.p).round(1).to_string())

# recalibrate: quantile-regression of y on p using 403, apply to 431
cal = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, learning_rate=0.05,
                   n_estimators=300, max_depth=3, min_child_weight=20, subsample=0.9,
                   colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist", n_jobs=8)
cal.fit(pc.to_frame("p"), yc)
pr = pd.Series(cal.predict(pe.to_frame("p")), index=pe.index).clip(lower=0)
print("recal MAE 431: %.2f (raw %.2f)" % (np.abs(pr-ye).mean(), np.abs(pe-ye).mean()))

# global multiplicative / additive checks on 431
for c in [0.95, 1.0, 1.05, 1.1, 1.15]:
    print("scale %.2f -> MAE %.2f" % (c, np.abs(pe*c-ye).mean()))
med = float(yc.median()); print("median y 403: %.2f" % med)
for a in [-5, 0, 5]:
    print("shift %+d -> MAE %.2f" % (a, np.abs(pe+a-ye).mean()))


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

base = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
preds = {}
t0=time.time()
m1 = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **base)
m1.fit(Xtr, ytr, sample_weight=w); preds["quant"] = m1.predict
print("quant %.0fs" % (time.time()-t0)); t0=time.time()

b2 = dict(base); b2["n_estimators"]=1200
m2 = XGBRegressor(objective="reg:absoluteerror", **b2)
m2.fit(Xtr, ytr, sample_weight=w); preds["l1"] = m2.predict
print("l1 %.0fs" % (time.time()-t0)); t0=time.time()

m3 = XGBRegressor(objective="reg:squarederror", **b2)
m3.fit(Xtr, np.log1p(ytr), sample_weight=w)
preds["log"] = lambda X: np.expm1(m3.predict(X))
print("log %.0fs" % (time.time()-t0))

P = {k: pd.Series(np.clip(f(Xe),0,None), index=Xe.index) for k,f in preds.items()}
for k,v in P.items():
    print("%-6s MAE431 %.2f  mean %.1f" % (k, np.abs(v-ye).mean(), v.mean()))
import itertools
keys = list(P)
for r in (2,3):
    for combo in itertools.combinations(keys, r):
        avg = sum(P[k] for k in combo)/r
        print("+".join(combo), "MAE431 %.2f" % np.abs(avg-ye).mean())
# median-of-3
print("median3 MAE431 %.2f" % np.abs(pd.concat(P.values(),axis=1).median(axis=1)-ye).mean())


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["house_key" if False else "household_key"], on_err=None) if False else F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 403.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

base = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n26=None) if False else dict(
            learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **base)
m.fit(Xtr, ytr, sample_weight=w)
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
pc = pd.Series(m.predict(Xc), index=Xc.index).clip(lower=0)
print("decay403 MAE431 %.2f  MAE403 %.2f" % (np.abs(pe-ye).mean(), np.abs(pc-yc).mean()))

# per-snapshot mean bias on 403
print("403 mean y %.1f mean p %.1f" % (yc.mean(), pc.mean()))


# ---- cell ----

import pandas as pd, numpy as np, time

def add_feats(view, s):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.day <= s]
    w = (s + 8) // 7
    out = pd.DataFrame(index=hh)

    # weekly spend series (last 52 weeks, selected offsets)
    tw = tx[(tx.week_no >= w-52) & (tx.week_no <= w-1)]
    pv = tw.pivot_table(index="household_key", columns="week_no", values="sales_value", aggfunc="sum")
    for k in [1,2,3,4,5,6,7,8,13,26,39,52]:
        col = w - k
        out[f"wk_{k}"] = pv[col].reindex(hh).fillna(0.0) if col in pv.columns else 0.0
    aw = (pv > 0).sum(axis=1).reindex(hh).fillna(0)
    out["wk_active_8"] = np.minimum(aw, 8)

    # shifted 28d windows (previous periods, same length)
    for nm, lo, hi in [("lag28_56", s-55, s-28), ("lag56_84", s-83, s-56), ("lag84_112", s-111, s-84),
                       ("sp28_yag", s-391, s-364), ("sp84_yag", s-447, s-364)]:
        m = tx[(tx.day >= lo) & (tx.day <= hi)]
        out[nm] = m.groupby("household_key").sales_value.sum().reindex(hh).fillna(0.0)

    # day-of-week spend shares (84d); dow = day % 7
    t84 = tx[tx.day > s-84]
    dow = t84.assign(dow=t84.day % 7).pivot_table(index="household_key", columns="dow",
                                                  values="sales_value", aggfunc="sum").reindex(hh).fillna(0.0)
    tot = dow.sum(axis=1)
    for d in range(7):
        out[f"dow_{d}"] = (dow[d] / tot.replace(0, np.nan)).fillna(0.0)
    # weekend share
    out["wkend_share"] = (dow.get(5, 0) + dow.get(6, 0)) / tot.replace(0, np.nan)
    out["wkend_share"] = out["wkend_share"].fillna(0.0)

    # store concentration (84d)
    st = t84.groupby(["household_key", "store_id"]).sales_value.sum()
    out["topstore_share_84"] = (st.groupby(level=0).max() / st.groupby(level=0).sum()).reindex(hh).fillna(0.0)
    # distinct commodities (84d)
    out["ncommod_84"] = t84.groupby("household_key").commodity_desc.nunique().reindex(hh).fillna(0) \
        if "commodity_desc" in t84.columns else 0.0
    return out

def fn(view, s):
    return add_feats(view, s)

t0 = time.time()
F2 = build_features(fn)
print("built", F2.shape, "%.0fs" % (time.time()-t0))
print(F2.head(3).T.round(2))
save_table(F2, "f_weekly.parquet")


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
F2 = load_saved("f_weekly.parquet").drop(columns=["index"], errors="ignore")
F = F.merge(F2, on=["household_key","snapshot_day"], how="left").fillna(F2.mean(numeric_only=True))
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

base = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
            subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **base)
m.fit(Xtr, ytr, sample_weight=w)
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
pc = pd.Series(m.predict(Xc), index=Xc.index).clip(lower=0)
print("with weekly feats: MAE431 %.2f  MAE403 %.2f" % (np.abs(pe-ye).mean(), np.abs(pc-yc).mean()))

imp = pd.Series(m.feature_importances_, index=feat).sort_values(ascending=False)
print("top gains:"); print(imp.head(15).round(4).to_string())
print("weekly feats gain sum:", imp[[c for c in imp.index if c.startswith(('wk_','lag','sp','dow','wkend','topstore','ncommod'))]].sum().round(4))


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key","snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key","snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key","snapshot_day")]
tt = train_targets().set_index(["household_key","snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key","snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95,123,151,179,207,235,263,291,319,347,375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

def fitq(**kw):
    b = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
    b.update(kw)
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **b)
    m.fit(Xtr, ytr, sample_weight=w)
    return m

t0=time.time()
m8 = fitq(max_depth=8, min_child_weight=20)
p8 = pd.Series(m8.predict(Xe), index=Xe.index).clip(lower=0)
print("depth8 MAE431 %.2f (%.0fs)" % (np.abs(p8-ye).mean(), time.time()-t0)); t0=time.time()

m36 = fitq(n_estimators=3600, learning_rate=0.02)
p36 = pd.Series(m36.predict(Xe), index=Xe.index).clip(lower=0)
print("3600@.02 MAE431 %.2f (%.0fs)" % (np.abs(p36-ye).mean(), time.time()-t0))

# clip caps on the depth-6 base prediction (recompute quickly with saved-style model? reuse p36/p8)
base = pd.Series(fitq().predict(Xe), index=Xe.index).clip(lower=0)
print("base MAE431 %.2f" % np.abs(base-ye).mean())
for cap in [500, 700, 900, 1100]:
    print("clip@%d -> %.2f" % (cap, np.abs(base.clip(upper=cap)-ye).mean()))
# blend with persistence
for al in [0.1, 0.2, 0.3]:
    per = Xe["seq_mean"] if "seq_mean" in Xe else Xe["spend_84"]
    bl = (1-al)*base + al*per.values
    print("blend seq_mean a=%.1f -> %.2f" % (al, np.abs(bl-ye).mean()))
# 2-seed ensemble
m_s2 = fitq(subsample=0.7, colsample_bytree=0.6)
p_s2 = pd.Series(m_s2.predict(Xe), index=Xe.index).clip(lower=0)
print("2-seed avg MAE431 %.2f" % np.abs(((base+p_s2)/2)-ye).mean())


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key", "snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key", "snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key", "snapshot_day")]
tt = train_targets().set_index(["household_key", "snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key", "snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95, 123, 151, 179, 207, 235, 263, 291, 319, 347, 375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431]); Xc, yc = mat([403])

def fitq(**kw):
    b = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
    b.update(kw)
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **b)
    m.fit(Xtr, ytr, sample_weight=w)
    return m

m = fitq()
pc = pd.Series(m.predict(Xc), index=Xc.index).clip(lower=0)
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
per = Xe["seq_mean"]
d = pd.DataFrame({"p": pe, "per": per.values, "y": ye})

for thr in [30, 50, 70, 100]:
    for al in [0.3, 0.5]:
        weak = d.p < thr
        bl = d.p.copy(); bl[weak] = (1 - al) * d.p[weak] + al * d.per[weak]
        print("thr %3d a=%.1f -> %.2f (n_weak %d)" % (thr, al, np.abs(bl - ye).mean(), weak.sum()))

weak = d.p < 50
print("MAE per on weak rows: %.2f vs model %.2f" % (np.abs(d.per[weak] - d.y[weak]).mean(), np.abs(d.p[weak] - d.y[weak]).mean()))


# ---- cell ----

import pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor

F = load_saved("e004_features.parquet")
f5n = load_saved("e005_newfeats.parquet")
c5 = [c for c in f5n.columns if c not in ("household_key", "snapshot_day")]
F = F.merge(f5n.drop(columns=[c for c in c5 if c in F.columns]), on=["household_key", "snapshot_day"], how="inner")
feat = [c for c in F.columns if c not in ("household_key", "snapshot_day")]
tt = train_targets().set_index(["household_key", "snapshot_day"]).future_spend_4w

def mat(days):
    X = F[F.snapshot_day.isin(days)].set_index(["household_key", "snapshot_day"])[feat]
    return X, tt.reindex(X.index)

Xtr, ytr = mat([95, 123, 151, 179, 207, 235, 263, 291, 319, 347, 375])
w = pd.Series(0.5 ** ((403 - Xtr.index.get_level_values(1)) / 140.0), index=Xtr.index)
Xe, ye = mat([431])

def fitq(**kw):
    b = dict(learning_rate=0.03, n_estimators=2400, max_depth=6, min_child_weight=10,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, tree_method="hist", n_jobs=8)
    b.update(kw)
    m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, **b)
    m.fit(Xtr, ytr, sample_weight=w)
    return m

m = fitq()
pe = pd.Series(m.predict(Xe), index=Xe.index).clip(lower=0)
per = Xe["seq_mean"]
pred = (0.8 * pe + 0.2 * per).clip(lower=0)
print("blend MAE431 %.2f" % np.abs(pred - ye).mean())

vd = snapshot_days()["validation"]
Xv, _ = mat(vd)
pv = pd.Series(m.predict(Xv), index=Xv.index).clip(lower=0)
predv = (0.8 * pv + 0.2 * Xv["seq_mean"]).clip(lower=0)
out = predv.reset_index()
out.columns = ["household_key", "snapshot_day", "prediction"]
p = save_table(out, "e009_preds.parquet")
print(p, out.shape, out.prediction.describe().round(2).to_dict())
