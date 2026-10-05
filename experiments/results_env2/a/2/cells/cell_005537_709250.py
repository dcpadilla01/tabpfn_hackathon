import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

oof = A.load_saved("oof_e008.parquet"); tt = A.train_targets(); feats = A.load_saved("feats_v3.parquet")
m = oof.merge(tt, on=["household_key","snapshot_day"]).merge(feats, on=["household_key","snapshot_day"])
y = m.future_spend_4w.values

# find blend weights approximating E008
best=None
for wm in [0.4,0.5,0.6]:
    for ws in [0.1,0.2,0.3,0.4]:
        wl = 1-wm-ws
        if wl<0: continue
        p = wm*m.oof_med + ws*m.oof_sq + wl*m.oof_log
        mae = np.abs(p-y).mean()
        if best is None or mae<best[0]: best=(mae,wm,ws,wl)
print("best oof blend:", best)
mae,wm,ws,wl = best
m["blend"] = wm*m.oof_med + ws*m.oof_sq + wl*m.oof_log

trm = m[m.snapshot_day<=347]; vam = m[m.snapshot_day>=375]
SF = ["blend","spend_28","baskets_28","active_28","days_since_last","avg_basket_84",
      "trend_28_56","week_of_year","spend_112","baskets_112","has_demo","spend_7","baskets_84"]
def stacker_fit(d):
    X = d[SF].copy()
    mod = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, min_child_weight=20,
                           subsample=0.8, colsample_bytree=0.8, objective="reg:quantileerror",
                           quantile_alpha=0.5, random_state=0, n_jobs=8)
    mod.fit(X, d.future_spend_4w)
    return mod
st = stacker_fit(trm)
p_st = st.predict(vam[SF])
p_raw = vam.blend.values; yv = vam.future_spend_4w.values
print("raw blend  holdout(375-431) MAE %.2f" % np.abs(p_raw-yv).mean())
print("stacked    holdout(375-431) MAE %.2f" % np.abs(p_st-yv).mean())
for sd in [375,403,431]:
    k = vam.snapshot_day==sd
    print("  snap",sd,"raw %.2f stacked %.2f" % (np.abs(p_raw[k.values]-yv[k.values]).mean(), np.abs(p_st[k.values]-yv[k.values]).mean()))

# simple decile ratio calibration
trm2 = trm.copy(); trm2["bin"] = pd.qcut(trm2.blend, 10, duplicates="drop")
ratio = trm2.groupby("bin", observed=True).apply(lambda d: np.median(d.future_spend_4w/np.maximum(d.blend,1)))
vam2 = vam.copy(); vam2["bin"] = pd.cut(vam2.blend, trm2["bin"].cat.categories)
p_cal = vam2.blend.values * vam2["bin"].map(ratio).astype(float).fillna(1).values
print("decile-ratio calib MAE %.2f" % np.abs(p_cal-yv).mean())
print("stacked bias %.1f raw bias %.1f" % ((p_st-yv).mean(), (p_raw-yv).mean()))
