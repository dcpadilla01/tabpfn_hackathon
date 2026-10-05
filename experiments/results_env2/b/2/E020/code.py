
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved("e011_table.parquet")
print("e011:", t.shape)
print(list(t.columns))
print()

for name in ["hazard_v1","churn_vol_v1","timing_v1","deal_v1","selfcal_v1","union_all_v1","twin_v1"]:
    try:
        d = agent_api.load_saved(name + ".parquet")
        print(name, d.shape, list(d.columns)[:12])
    except Exception as e:
        print(name, "ERR", type(e).__name__, e)
print()
tt = agent_api.train_targets()
print("targets:", tt.shape, tt.columns.tolist())
k1 = set(map(tuple, t[["household_key","snapshot_day"]].values))
k2 = set(map(tuple, tt[["household_key","snapshot_day"]].values))
print("e011 keys == train keys?", k1==k2, len(k1), len(k2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
print("dup keys hazard:", h[key].duplicated().sum(), "| hazard dtypes ok:", h.drop(columns=key).dtypes.unique())

m = e.merge(h, on=key, how="inner")
print("merged:", m.shape, "== e011 rows:", len(e))
print("hazard cols:", [c for c in h.columns if c not in key])

tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")
tr = mm["future_spend_4w"].notna().values
print("train rows:", tr.sum(), "val rows:", (~tr).sum())

# quick proxy: ridge on standardized features, fit train -> MAE val
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = mm[feats].astype(float).values
X = np.nan_to_num(X, nan=0.0)
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
Xs = (X - mu) / sd
y = mm["future_spend_4w"].values

def ridge_eval(cols_mask, alpha=100.0, logt=False):
    A = Xs[tr][:, cols_mask]; yy = y[tr]
    if logt: yy = np.log1p(yy)
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = Xs[~tr][:, cols_mask] @ w
    if logt: p = np.expm1(p)
    p = np.clip(p, 0, None)
    return np.abs(p - y[~tr]).mean()

i28 = feats.index("spend_28d")
print("proxy ridge MAE  spend_28d only:", round(ridge_eval(np.eye(len(feats))[i28].astype(bool)),2))
allm = np.ones(len(feats), bool)
hz = np.array([c in [c for c in h.columns if c not in key] for c in feats])
print("proxy ridge MAE  E011(all):", round(ridge_eval(allm),2))
print("proxy ridge MAE  E011+hazard:", round(ridge_eval(allm | hz),2))
print("proxy ridge MAE  E011+hazard (logt):", round(ridge_eval(allm | hz, logt=True),2))
print("proxy ridge MAE  E011 (logt):", round(ridge_eval(allm, logt=True),2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")

feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values

tr_fit = day <= 403   # pseudo-train
tr_val = day == 431   # pseudo-val (last train snapshot)
print("fit rows:", tr_fit.sum(), "val rows:", tr_val.sum())

mu, sd = X[tr_fit].mean(0), X[tr_fit].std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(mask, alpha=100.0, logt=False, w=None):
    A, yy = Xs[tr_fit][:, mask], y[tr_fit]
    if logt: yy = np.log1p(yy)
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = Xs[tr_val][:, mask] @ w
    if logt: p = np.expm1(p)
    p = np.clip(p, 0, None)
    return np.abs(p - y[tr_val]).mean()

hz = np.array([c in [c for c in h.columns if c not in key] for c in feats])
allm = np.ones(len(feats), bool)
for a in [30, 100, 300]:
    print(f"alpha={a}: E011={ridge_eval(allm,a):.2f}  E011+hz={ridge_eval(allm|hz,a):.2f}")
print("logt: E011=", round(ridge_eval(allm,100,logt=True),2), " E011+hz=", round(ridge_eval(allm|hz,100,logt=True),2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
hz_cols = [c for c in h.columns if c not in key]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")

feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
tr_fit = day <= 403; tr_val = day == 431

mu, sd = X[tr_fit].mean(0), X[tr_fit].std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, alpha=100.0):
    idx = [feats.index(c) for c in cols]
    A, yy = Xs[tr_fit][:, idx], y[tr_fit]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[tr_val][:, idx] @ w, 0, None)
    return np.abs(p - y[tr_val]).mean()

e011_cols = [c for c in e.columns if c not in key]
print("n e011 feats:", len(e011_cols))
for a in [30, 100, 300]:
    print(f"alpha={a}: E011={ridge_eval(e011_cols,a):.3f}  +hazard={ridge_eval(e011_cols+hz_cols,a):.3f}")

# explicit interaction candidates
mm["x_spd_sc"] = mm["spend_28d"] * mm["sc_ratio_mean"].fillna(1.0)
mm["x_ew_sc"]  = mm["ew_spend_hl28"] * mm["sc_ratio_mean"].fillna(1.0)
mm["x_spd_carry"] = mm["spend_28d"] * mm["sc_carry"].fillna(1.0)
mm["x_usual_sc"] = mm["usual_4w"] * mm["sc_ratio_mean"].fillna(1.0)
inter = ["x_spd_sc","x_ew_sc","x_spd_carry","x_usual_sc"]
for a in [100]:
    print(f"alpha={a}: +hazard+inter={ridge_eval(e011_cols+hz_cols+inter,a):.3f}  +inter={ridge_eval(e011_cols+inter,a):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
hz_cols = [c for c in h.columns if c not in key]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")

for c, v in [("x_spd_sc", mm["spend_28d"]*mm["sc_ratio_mean"].fillna(1.0)),
             ("x_ew_sc", mm["ew_spend_hl28"]*mm["sc_ratio_mean"].fillna(1.0)),
             ("x_spd_carry", mm["spend_28d"]*mm["sc_carry"].fillna(1.0)),
             ("x_usual_sc", mm["usual_4w"]*mm["sc_ratio_mean"].fillna(1.0))]:
    mm[c] = v

feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
tr_fit = day <= 403; tr_val = day == 431
mu, sd = X[tr_fit].mean(0), X[tr_fit].std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, alpha=100.0):
    idx = [feats.index(c) for c in cols]
    A, yy = Xs[tr_fit][:, idx], y[tr_fit]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[tr_val][:, idx] @ w, 0, None)
    return np.abs(p - y[tr_val]).mean()

e011_cols = [c for c in e.columns if c not in key]
print("base:", round(ridge_eval(e011_cols),3))
print("+hz :", round(ridge_eval(e011_cols+hz_cols),3))
print("+hz+inter:", round(ridge_eval(e011_cols+hz_cols+["x_spd_sc","x_ew_sc","x_spd_carry","x_usual_sc"]),3))
print("+inter:", round(ridge_eval(e011_cols+["x_spd_sc","x_ew_sc","x_spd_sc","x_usual_sc"]),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
h = agent_api.load_saved("hazard_v1.parquet")
key = ["household_key","snapshot_day"]
hz_cols = [c for c in h.columns if c not in key]
m = e.merge(h, on=key, how="inner")
tt = agent_api.train_targets()
mm = m.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=100.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

e011_cols = [c for c in e.columns if c not in key]
pairs = [(375,403),(403,431),(347,375)]
for a in [30,100]:
    d0, d1 = [], []
    for f,v in pairs:
        d0.append(ridge_eval(e011_cols, f, v, a))
        d1.append(ridge_eval(e011_cols+hz_cols, f, v, a))
    print(f"alpha={a}  E011: {[round(x,2) for x in d0]} avg={np.mean(d0):.3f}")
    print(f"alpha={a}  +hz : {[round(x,2) for x in d1]} avg={np.mean(d1):.3f}  delta={np.mean(d1)-np.mean(d0):+.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
blocks = {}
for name in ["hazard_v1","deal_v1","timing_v1","churn_vol_v1"]:
    try:
        d = agent_api.load_saved(name + ".parquet")
        cols = [c for c in d.columns if c not in key]
        blocks[name] = d[key + cols]
        print(name, "cols:", cols)
    except Exception as ex:
        print(name, "ERR", ex)

tt = agent_api.train_targets()
mm = e.merge(tt, on=key, how="left")
for n, d in blocks.items():
    mm = mm.merge(d, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

e011_cols = [c for c in e.columns if c not in key]
combos = {
    "E011": e011_cols,
    "E011+hz": e011_cols + [c for c in blocks["hazard_v1"].columns if c not in key],
    "E011+deal": e011_cols + [c for c in blocks["deal_v1"].columns if c not in key],
    "E011+timing": e011_cols + [c for c in blocks["timing_v1"].columns if c not in key],
    "E011+hz+deal": e011_cols + [c for c in blocks["hazard_v1"].columns if c not in key] + [c for c in blocks["deal_v1"].columns if c not in key],
}
pairs = [(375,403),(403,431)]
for cname, cols in combos.items():
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:14s} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
e011_cols = [c for c in e.columns if c not in key]
base = e.copy()

def newcols(d):
    return [c for c in d.columns if c not in key and c not in e011_cols]

blocks = {}
for name in ["hazard_v1","deal_v1","timing_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    nc = newcols(d)
    blocks[name] = nc
    base = base.merge(d[key+nc], on=key, how="left")
print({k: len(v) for k,v in blocks.items()})

tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

combos = {
    "E011": e011_cols,
    "E011+hz": e011_cols + blocks["hazard_v1"],
    "E011+deal": e011_cols + blocks["deal_v1"],
    "E011+timing": e011_cols + blocks["timing_v1"],
    "E011+hz+deal+timing": e011_cols + blocks["hazard_v1"] + blocks["deal_v1"] + blocks["timing_v1"],
}
pairs = [(375,403),(403,431)]
for cname, cols in combos.items():
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:22s} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
e011_cols = [c for c in e.columns if c not in key]
base = e.copy()
def newcols(d):
    return [c for c in d.columns if c not in key and c not in e011_cols]

d = agent_api.load_saved("display_v1.parquet")
print("display_v1 cols:", newcols(d))
hz = agent_api.load_saved("hazard_v1.parquet")
nc_hz = newcols(hz)
base = base.merge(d[key+newcols(d)], on=key, how="left").merge(hz[key+nc_hz], on=key, how="left")

tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

disp = newcols(d)
combos = {
    "E011": e011_cols,
    "E011+disp": e011_cols + disp,
    "E011+hz": e011_cols + nc_hz,
    "E011+disp+hz": e011_cols + disp + nc_hz,
}
pairs = [(375,403),(403,431)]
for cname, cols in combos.items():
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:14s} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def make_cal2(view, D):
    tx = view.table("transactions")
    hh = view.households
    try: hh = list(hh)
    except: pass
    tx = tx[tx["household_key"].isin(set(hh))]
    rows = {}
    for h, sub in tx.groupby("household_key", sort=False):
        days = sub["day"].values.astype(float)
        sp = sub["sales_value"].values.astype(float)
        o = np.argsort(days, kind="stable")
        days, sp = days[o], sp[o]
        first = days[0]
        cs = np.concatenate([[0.0], np.cumsum(sp)])
        def wsum(a, b):
            i = np.searchsorted(days, a, "left"); j = np.searchsorted(days, b, "right")
            return cs[j] - cs[i]
        Tcur = wsum(D-27, D)
        ts, Ts, Fs = [], [], []
        t = D - 28
        while t >= first + 84:
            Ts.append(wsum(t-27, t)); Fs.append(wsum(t+1, t+28)); ts.append(t)
            t -= 28
        n = len(ts)
        d = dict(c2_n=float(n), c2_tcur=Tcur)
        if n:
            T = np.array(Ts); F = np.array(Fs)
            fm, fmed, fsd = F.mean(), np.median(F), F.std()
            d.update(c2_f_mean=fm, c2_f_med=fmed, c2_f_std=fsd, c2_t_mean=T.mean())
            w = 0.5 ** np.arange(n)          # most recent pair weight 1
            d["c2_f_ew"] = float((F*w).sum()/w.sum())
            d["c2_ratio_ew"] = float((F*w).sum()/max((T*w).sum(), 1e-9))
            d["c2_ratio_last"] = float(F[-1]/max(T[-1], 1e-9))
            m0 = T <= 1e-9
            d["c2_f_zero"] = float(F[m0].mean()) if m0.any() else np.nan
            d["c2_n_zero"] = float(m0.sum())
            if n >= 2 and T.var() > 1e-9:
                sl, ic = np.polyfit(T, F, 1)
                k = n/(n+3.0)
                sl_sh = sl*k
                ic_sh = F.mean() - sl_sh*T.mean()
                d["c2_slope"] = float(sl_sh); d["c2_ic"] = float(ic_sh)
                d["c2_pred"] = float(ic_sh + sl_sh*Tcur)
                d["c2_resid"] = float((F - (ic+sl*T)).std())
                d["c2_corr"] = float(np.corrcoef(T, F)[0,1])
                m = min(n, 6)
                if np.var(T[-m:]) > 1e-9:
                    sl6, ic6 = np.polyfit(T[-m:], F[-m:], 1)
                    d["c2_slope6"] = float(sl6*k)
                    d["c2_pred6"] = float(ic6 + sl6*Tcur)
        rows[h] = d
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "household_key"
    return df.reindex(hh)

tab = agent_api.build_features(make_cal2)
print(tab.shape)
print(tab.columns.tolist())
print(tab.isna().mean().round(3).to_dict())
p = agent_api.save_table(tab, "cal2_v1")
print("saved:", p)


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
c2 = agent_api.load_saved("cal2_v1.parquet")
key = ["household_key","snapshot_day"]
c2_cols = [c for c in c2.columns if c not in key]
base = e.merge(c2, on=key, how="left")
tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

e011_cols = [c for c in e.columns if c not in key]
combos = {
    "E011": e011_cols,
    "E011+c2": e011_cols + c2_cols,
    "c2 only": c2_cols,
}
pairs = [(375,403),(403,431)]
for cname, cols in combos.items():
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:10s} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
e011_cols = [c for c in e.columns if c not in key]
base = e.copy()
def newcols(d):
    return [c for c in d.columns if c not in key and c not in e011_cols]

adds = {}
for name in ["hazard_v1","cal2_v1","deal_v1","timing_v1","display_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    adds[name] = newcols(d)
    base = base.merge(d[key+adds[name]], on=key, how="left")
print({k: len(v) for k,v in adds.items()}, "total feats:", len(e011_cols)+sum(len(v) for v in adds.values()))

tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

combos = {
    "E011": e011_cols,
    "E011+hz": e011_cols + adds["hazard_v1"],
    "E011+hz+c2": e011_cols + adds["hazard_v1"] + adds["cal2_v1"],
    "E011+ALL": e011_cols + sum(adds.values(), []),
}
pairs = [(375,403),(403,431)]
for cname, cols in combos.items():
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:12s} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd, numpy as np

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
e011_cols = [c for c in e.columns if c not in key]
base = e.copy()
def newcols(d):
    return [c for c in d.columns if c not in key and c not in e011_cols]
adds = {}
for name in ["hazard_v1","cal2_v1","deal_v1","timing_v1","display_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    adds[name] = newcols(d)
    base = base.merge(d[key+adds[name]], on=key, how="left")

tt = agent_api.train_targets()
mm = base.merge(tt, on=key, how="left")
feats = [c for c in mm.columns if c not in key + ["future_spend_4w"]]
X = np.nan_to_num(mm[feats].astype(float).values, nan=0.0)
y = mm["future_spend_4w"].values
day = mm["snapshot_day"].values
mu, sd = X.mean(0), X.std(0) + 1e-9
Xs = (X - mu) / sd

def ridge_eval(cols, fit_day_max, val_day, alpha=30.0):
    idx = [feats.index(c) for c in cols]
    trf = day <= fit_day_max; trv = day == val_day
    A, yy = Xs[trf][:, idx], y[trf]
    w = np.linalg.solve(A.T@A + alpha*np.eye(A.shape[1]), A.T@yy)
    p = np.clip(Xs[trv][:, idx] @ w, 0, None)
    return np.abs(p - y[trv]).mean()

pairs = [(375,403),(403,431)]
base_cols = e011_cols + adds["hazard_v1"] + adds["cal2_v1"]
# drop-noise test: add deal/timing/display but drop the weakest c2 cols (high NaN, near-duplicates)
drop_sets = {
    "hz+c2": [],
    "hz+c2-tiny": ["c2_f_zero","c2_n_zero","c2_ratio_last","c2_f_med","c2_f_std","c2_corr","c2_resid"],
    "hz+c2-tiny+deal": None,  # filled below
}
cols2 = base_cols + [c for c in adds["deal_v1"] if not c.startswith("redemp")]
drop_sets["hz+c2-tiny+deal"] = drop_sets["hz+c2-tiny"]
for cname, drop in drop_sets.items():
    cols = [c for c in cols2 if c not in drop]
    vals = [ridge_eval(cols, f, v) for f, v in pairs]
    print(f"{cname:18s} n={len(cols):3d} {[round(x,2) for x in vals]} avg={np.mean(vals):.3f}")


# ---- cell ----

import agent_api, pandas as pd

e = agent_api.load_saved("e011_table.parquet")
key = ["household_key","snapshot_day"]
out = e.copy()
for name in ["hazard_v1","cal2_v1"]:
    d = agent_api.load_saved(name + ".parquet")
    nc = [c for c in d.columns if c not in key and c not in out.columns]
    out = out.merge(d[key+nc], on=key, how="left")
print("final shape:", out.shape, "| dups:", out[key].duplicated().sum(), "| dtypes:", set(out.dtypes.astype(str)))
p = agent_api.save_table(out, "e020_final_v1")
print("saved:", p)
