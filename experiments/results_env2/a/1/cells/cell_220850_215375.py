import numpy as np, pandas as pd, xgboost as xgb

full = agent_api.load_saved("e004_features.parquet")
tt = agent_api.train_targets()
data = full.merge(tt, on=["household_key", "snapshot_day"])
data = data.sort_values(["snapshot_day", "household_key"]).reset_index(drop=True)
cols = [c for c in full.columns if c not in ("household_key", "snapshot_day")]
y = data.future_spend_4w.values
sd = data.snapshot_day.values

def wmedian(v, w):
    o = np.argsort(v); v, w = np.asarray(v)[o], np.asarray(w)[o]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]

def mk(**kw):
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, n_estimators=800,
             learning_rate=0.05, max_depth=6, min_child_weight=10, subsample=0.8,
             colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0)
    p.update(kw); return xgb.XGBRegressor(**p)

def run(train_mask, val_mask, variant):
    w = 0.5 ** ((431 - sd[train_mask]) / 277.0)
    Xtr, Xte = data.loc[train_mask, cols], data.loc[val_mask, cols]
    ytr = y[train_mask]
    if variant == "base":
        m = mk(base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    elif variant == "snapday":
        Xtr2 = Xtr.copy(); Xtr2["snap"] = sd[train_mask]
        Xte2 = Xte.copy(); Xte2["snap"] = sd[val_mask]
        m = mk(base_score=float(wmedian(ytr, w)))
        m.fit(Xtr2, ytr, sample_weight=w); p = m.predict(Xte2)
    elif variant == "log":
        m = mk(base_score=float(np.median(np.log1p(ytr))))
        m.fit(Xtr, np.log1p(ytr), sample_weight=w); p = np.expm1(m.predict(Xte))
    elif variant == "twopart":
        z = (ytr > 0).astype(int)
        mc = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                               n_jobs=-1, random_state=0)
        mc.fit(Xtr, z, sample_weight=w)
        pos = ytr > 0
        m = mk(base_score=float(wmedian(ytr[pos], w[pos])))
        m.fit(Xtr[pos], ytr[pos], sample_weight=w[pos])
        p = m.predict(Xte) * mc.predict_proba(Xte)[:, 1]
    elif variant == "deep":
        m = mk(max_depth=8, min_child_weight=20, n_estimators=1200, learning_rate=0.04,
               base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    elif variant == "shallow":
        m = mk(max_depth=4, n_estimators=1500, learning_rate=0.04,
               base_score=float(wmedian(ytr, w)))
        m.fit(Xtr, ytr, sample_weight=w); p = m.predict(Xte)
    return np.clip(p, 0, None)

for vs in [(403,), (431,), (403, 431)]:
    tr = data.snapshot_day <= (vs[0] - 28)
    va = data.snapshot_day.isin(vs)
    yv = y[va.values]
    line = [f"val{vs}"]
    for variant in ["base", "snapday", "log", "twopart", "deep", "shallow"]:
        p = run(tr.values, va.values, variant)
        line.append(f"{variant}={np.abs(p - yv).mean():.2f}")
    print("  ".join(line))
# naive: predict spend_28
for vs in [(403,), (431,)]:
    va = data.snapshot_day.isin(vs)
    print(f"naive spend_28 val{vs}:", np.abs(data.loc[va, "spend_28"].values - y[va.values]).mean())