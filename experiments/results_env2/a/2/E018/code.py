
import agent_api, pandas as pd, numpy as np
names = ["feats_all_e016","feats_e014","feats_prof","feats_seasonal","feats_v1","feats_v2","feats_v3","feats_v4","feats_v5","feats_v6",
         "oof_e008","oof_e016_cv","oof_harness","oof_pt","past_targets",
         "pred_e008","pred_e009","pred_e010","pred_e010_blend","pred_e014","pred_e016","pred_log1p","pred_prof","pred_pt","pred_seasonal"]
for n in names:
    try:
        df = agent_api.load_saved(n + ".parquet")
        print("==", n, df.shape, list(df.columns))
        print(df.head(2).to_string())
    except Exception as e:
        print("==", n, "ERR", type(e).__name__, e)
print("snapshot_days:", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("train_targets:", tt.shape, tt.columns.tolist())
print(tt.head(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
for n in ["oof_e008","oof_e016_cv","oof_harness","oof_pt"]:
    df = agent_api.load_saved(n + ".parquet")
    print("==", n, df.shape, df.columns.tolist())
    print(df.groupby("snapshot_day")["oof"].agg(["count","mean"]).to_string())
tt = agent_api.train_targets()
m = tt.merge(agent_api.load_saved("oof_e016_cv.parquet"), on=["household_key","snapshot_day"], how="left")
print("oof_e016_cv merged:", m["oof"].notna().mean())
print(m.dropna().groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g.oof).mean()))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
for n in ["oof_e008","oof_e016_cv","oof_harness","oof_pt"]:
    df = agent_api.load_saved(n + ".parquet")
    print("==", n, df.shape, df.columns.tolist())
    print(df.groupby("snapshot_day").agg({"oof_sq":"mean","oof_med":"mean","oof_log":"mean"}).to_string() if "oof_sq" in df.columns else
          df.groupby("snapshot_day")["oof"].agg(["count","mean"]).to_string())
tt = agent_api.train_targets()
m = tt.merge(agent_api.load_saved("oof_e016_cv.parquet"), on=["household_key","snapshot_day"], how="left")
print("oof_e016_cv merged:", m["oof"].notna().mean())
print(m.dropna(subset=["oof"]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g.oof).mean()))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
def mae(df, col):
    m = tt.merge(df, on=["household_key","snapshot_day"], how="inner")
    return m.dropna(subset=[col]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g[col]).mean())
cv = agent_api.load_saved("oof_e016_cv.parquet")
for c in ["med_v3","med_all","hgbq_v3","hgbq_all"]:
    m = tt.merge(cv[["household_key","snapshot_day",c]], on=["household_key","snapshot_day"])
    print(c, "MAE", np.abs(m[c]-m.future_spend_4w).mean().round(3), "n", m[c].notna().sum())
    print(m.dropna(subset=[c]).groupby("snapshot_day").apply(lambda g: np.abs(g.future_spend_4w-g[c]).mean()).round(2).to_dict())
# blend weights search on OOF
cv = cv.merge(tt, on=["household_key","snapshot_day"]).dropna()
y = cv.future_spend_4w.values
best=None
for wm in np.linspace(0,1,21):
    for wv in np.linspace(0,1,21):
        p = wm*cv.med_v3.values + wv*cv.hgbq_v3.values + (1-wm-wv)*cv.med_all.values
        if (1-wm-wv)<-0.001: continue
        e = np.abs(p-y).mean()
        if best is None or e<best[0]: best=(e,wm,wv)
print("best OOF blend med_v3/hgbq_v3/med_all:", best)
for wm in np.linspace(0,1,21):
    for wv in np.linspace(0,1,21):
        p = wm*cv.med_v3.values + wv*cv.hgbq_all.values + (1-wm-wv)*cv.med_all.values
        if (1-wm-wv)<-0.001: continue
        e = np.abs(p-y).mean()
        if best is None or e<best[0]: best=(e,wm,wv,"v3/all")
print("best overall:", best)


# ---- cell ----

import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","count"])
print(g.to_string())
# trend check: linear fit of mean spend vs snapshot day
d = g.reset_index()
z = np.polyfit(d.snapshot_day, d["mean"], 1)
print("slope per 28d:", z[0], "intercept:", z[1])
med = tt.groupby("snapshot_day").future_spend_4w.median().reset_index()
z2 = np.polyfit(med.snapshot_day, med.future_spend_4w, 1)
print("median slope:", z2[0], z2[1])
# OOF bias by day
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
for c in ["med_v3","hgbq_all"]:
    cv["e_"+c] = cv[c]-cv.future_spend_4w
print(cv.groupby("snapshot_day")[["e_med_v3","e_hgbq_all"]].mean().round(2).to_string())
# E016 exact OOF blend
e016_oof = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
print("E016 blend OOF MAE:", np.abs(e016_oof-cv.future_spend_4w).mean().round(4))
print("bias of E016 blend:", (e016_oof-cv.future_spend_4w).mean().round(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
# per-snapshot optimal constant shift
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
g = cv.groupby("snapshot_day").apply(lambda x: pd.Series({
    "shift": (x.future_spend_4w-x.blend).mean(),
    "scale": (x.future_spend_4w*x.blend).sum()/(x.blend*x.blend).sum(),
    "mae0": np.abs(x.blend-x.future_spend_4w).mean()}))
print(g.round(3).to_string())
# global scale on OOF
for s in [1.0,1.05,1.1,1.15,1.2,1.25]:
    print(s, np.abs(s*cv.blend-cv.future_spend_4w).mean().round(3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
pop = feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean"),
                                        pop_mean_s112=("spend_112","mean"), n_hh=("household_key","count"))
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}))
d = sh.join(pop).reset_index()
print(d.round(3).to_string())
print("corr shift vs pop_mean_s28:", d.shift.corr(d.pop_mean_s28).round(3))
print("corr shift vs pop_mean_s84:", d.shift.corr(d.pop_mean_s84).round(3))
print("corr shift vs day:", d.shift.corr(d.snapshot_day).round(3))
print("corr shift vs n_hh:", d.shift.corr(d.n_hh).round(3))
# OLS shift ~ day
z = np.polyfit(d.snapshot_day, d.shift, 1); print("shift~day slope:", z)
# residual after day fit
pred_sh = np.polyval(z, d.snapshot_day); print("resid std:", (d.shift-pred_sh).std().round(2), "mean abs:", np.abs(d.shift-pred_sh).mean().round(2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
pop = feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean"),
                                        pop_mean_s112=("spend_112","mean"), n_hh=("household_key","count")).reset_index()
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(pop, on="snapshot_day")
print(d.round(3).to_string())
print("corr shift vs pop_mean_s28:", d["shift"].corr(d["pop_mean_s28"]).round(3))
print("corr shift vs pop_mean_s84:", d["shift"].corr(d["pop_mean_s84"]).round(3))
print("corr shift vs day:", d["shift"].corr(d["snapshot_day"]).round(3))
z = np.polyfit(d.snapshot_day, d["shift"], 1); print("shift~day fit:", z)
pred_sh = np.polyval(z, d.snapshot_day)
print("resid std:", (d["shift"]-pred_sh).std().round(2), "meanabs:", np.abs(d["shift"]-pred_sh).mean().round(2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean"),
        pop_mean_s112=("spend_112","mean"), pop_mean_s224=("spend_224","mean"),
        pop_mean_s56=("spend_56","mean"), pop_mean_b28=("baskets_28","mean")).reset_index(), on="snapshot_day")
# regression: shift ~ a + b*pop_mean_s28 + c*pop_mean_s84 (+ day)
X = np.column_stack([np.ones(len(d)), d.pop_mean_s28, d.pop_mean_s84, d.snapshot_day])
for cols,label in [([0,1],"s28"),([0,1,2],"s28+s84"),([0,1,2,3],"+day"),([0,3],"day only"),([0,2],"s84 only")]:
    b, *_ = np.linalg.lstsq(X[:,cols], d["shift"].values, rcond=None)
    r = d["shift"].values - X[:,cols]@b
    print(label, "coef", b.round(4), "resid std", r.std().round(2), "meanabs", np.abs(r).mean().round(2))


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean")).reset_index(), on="snapshot_day")
X = np.column_stack([np.ones(len(d)), d.pop_mean_s28, d.pop_mean_s84])
b,*_ = np.linalg.lstsq(X, d["shift"].values, rcond=None)
d["sh_pred"] = X@b
print(d[["snapshot_day","shift","sh_pred"]].round(2).to_string())
print("resid:", (d["shift"]-d.sh_pred).round(2).tolist())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
# per-snapshot mean of household-level 'median-ish' features vs realized mean target
d = feats.groupby("snapshot_day").agg(pop_med_s28=("spend_28","median"), pop_mean_s28=("spend_28","mean"),
                                      pop_mean_s84=("spend_84","mean"), pop_mean_s112=("spend_112","mean")).reset_index()
d = d.merge(tt.groupby("snapshot_day").future_spend_4w.agg(t_mean="mean", t_med="median").reset_index(), on="snapshot_day")
print(d.round(2).to_string())
print("corr t_mean vs pop_mean_s28:", d.t_mean.corr(d.pop_mean_s28).round(3))
print("corr t_mean vs pop_mean_s84:", d.t_mean.corr(d.pop_mean_s84).round(3))
print("corr t_mean vs pop_mean_s112:", d.t_mean.corr(d.pop_mean_s112).round(3))
# ratio t_mean / pop_mean_s28
print("ratio t_mean/pop_mean_s28:", (d.t_mean/d.pop_mean_s28).round(3).tolist())


# ---- cell ----

import agent_api, pandas as pd, numpy as np
feats = agent_api.load_saved("feats_all_e016.parquet")
tt = agent_api.train_targets()
cv = agent_api.load_saved("oof_e016_cv.parquet").merge(tt, on=["household_key","snapshot_day"])
cv["blend"] = 0.4*cv.med_v3 + 0.3*cv.hgbq_v3 + 0.3*cv.hgbq_all
sh = cv.groupby("snapshot_day").apply(lambda x: pd.Series({"shift":(x.future_spend_4w-x.blend).mean()}), include_groups=False).reset_index()
d = sh.merge(feats.groupby("snapshot_day").agg(pop_mean_s28=("spend_28","mean"), pop_mean_s84=("spend_84","mean")).reset_index(), on="snapshot_day")
X = np.column_stack([np.ones(len(d)), d.pop_mean_s28, d.pop_mean_s84])
b,*_ = np.linalg.lstsq(X, d["shift"].values, rcond=None)
d["sh_pred"] = X@b
# apply to validation predictions
p = agent_api.load_saved("pred_e016.parquet")
pv = p.merge(feats[["household_key","snapshot_day","spend_28","spend_84"]].rename(columns={"spend_28":"s28","spend_84":"s84"}), on=["household_key","snapshot_day"])
pv["sh"] = b[0] + b[1]*pv.s28 + b[2]*pv.s84
pv["prediction"] = (pv.prediction + pv.sh).clip(lower=0)
out = pv[["household_key","snapshot_day","prediction"]]
print(out.shape, out.snapshot_day.value_counts().to_dict())
agent_api.save_table(out, "pred_e018")
