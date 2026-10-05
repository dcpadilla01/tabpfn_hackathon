import agent_api, pandas as pd, numpy as np
pd.set_option("display.width", 200)
print(agent_api.snapshot_days())
for name in ["feats_v3","feats_seasonal"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape)
    print(list(df.columns))
for name in ["pred_seasonal","pred_e008","pred_e010_blend","oof_e008","pred_e009"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape, list(df.columns))
    print(df.head(3))
tt = agent_api.train_targets()
print("== train_targets", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
import xgboost, sklearn
print("xgb", xgboost.__version__, "sklearn", sklearn.__version__)


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet")
print(oof.snapshot_day.value_counts().sort_index())
tt = agent_api.train_targets()
m = oof.merge(tt, on=["household_key","snapshot_day"])
print(m.shape)
for c in ["oof_sq","oof_med","oof_log"]:
    print(c, "MAE", np.abs(m[c]-m.future_spend_4w).mean())
# per-snapshot-day bias
m["resid_med"] = m.future_spend_4w - m.oof_med
print(m.groupby("snapshot_day").agg(n=("resid_med","size"), mae=("resid_med", lambda s: s.abs().mean()), bias=("resid_med","mean"), tgt=("future_spend_4w","mean")))
# correlations among saved val predictions
preds = {n: agent_api.load_saved(n+".parquet") for n in ["pred_seasonal","pred_e008","pred_e009","pred_e010_blend","pred_log1p"]}
base = preds["pred_seasonal"][["household_key","snapshot_day"]].copy()
for n,p in preds.items():
    base = base.merge(p.rename(columns={"prediction":n}), on=["household_key","snapshot_day"])
print(base.shape)
print(base.drop(columns=["household_key","snapshot_day"]).corr().round(4))
print(base.drop(columns=["household_key","snapshot_day"]).describe().round(2))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
tt = agent_api.train_targets()
g = tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median","size"])
print(g.round(1))
oof = agent_api.load_saved("oof_e008.parquet").merge(tt, on=["household_key","snapshot_day"])
res = oof.future_spend_4w - oof.oof_med
print("median resid:", res.median(), "mean resid:", res.mean())
# what shift minimizes MAE of oof_med?
best = min(np.arange(-40,41,2), key=lambda s: np.abs(res-s).mean())
print("best shift on oof:", best, "MAE:", np.abs(res-best).mean(), "vs base:", np.abs(res).mean())
# week mapping
for d in [95,123,151,179,207,235,263,291,319,347,375,403,431,459,487,515,543]:
    print(d, (d+8)//7, end="  | ")
print()
# check week_of_year feature values in feats_v3
fv3 = agent_api.load_saved("feats_v3.parquet")
print(fv3.groupby("snapshot_day").week_of_year.first())


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet").merge(agent_api.train_targets(), on=["household_key","snapshot_day"])
oof["pred"] = oof.oof_med
oof["resid"] = oof.future_spend_4w - oof.pred
# residual vs prediction decile
oof["pb"] = pd.qcut(oof.pred, 10, duplicates="drop")
t = oof.groupby("pb", observed=True).agg(n=("resid","size"), pred=("pred","mean"), tgt=("future_spend_4w","median"), med_resid=("resid","median"), mae=("resid", lambda s: s.abs().mean()))
print(t.round(1))
# per-day optimal shift
for d, gr in oof.groupby("snapshot_day"):
    r = gr.resid.values
    shifts = np.arange(-30,31,2)
    maes = [np.abs(r-s).mean() for s in shifts]
    b = shifts[int(np.argmin(maes))]
    print(d, "base MAE %.2f"%np.abs(r).mean(), "best shift", b, "MAE %.2f"%min(maes))
# binned median calibration lookup (train-like fit on first 3 oof days, eval on day 431)
tr = oof[oof.snapshot_day<=403]; te = oof[oof.snapshot_day==431]
bins = np.quantile(tr.pred, np.linspace(0,1,21))
tr["b"] = pd.cut(tr.pred, bins, include_lowest=True)
lut = tr.groupby("b", observed=True).future_spend_4w.median()
def calib(p):
    idx = pd.cut(pd.Series(p), bins, include_lowest=True).cat.categories
    lab = pd.cut(pd.Series(p), bins, include_lowest=True)
    return lab.map(lut).astype(float).fillna(p).values
pc = calib(te.pred.values)
print("day431 raw MAE %.3f -> calibrated MAE %.3f"%(np.abs(te.resid).mean(), np.abs(te.future_spend_4w-pc).mean()))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
oof = agent_api.load_saved("oof_e008.parquet").merge(agent_api.train_targets(), on=["household_key","snapshot_day"])
oof["resid"] = oof.future_spend_4w - oof.oof_med
# 1) capping predictions
for cap in [400, 600, 800, 1000, 1200, 1500, np.inf]:
    print("cap", cap, "MAE %.3f"%np.minimum(oof.resid+0,0).add(0).pipe(lambda s: None) if False else np.abs(np.minimum(oof.oof_med, cap)-oof.future_spend_4w).mean())
# 2) binned-median calibration: fit on days<=403, eval 431
tr = oof[oof.snapshot_day<=403].copy(); te = oof[oof.snapshot_day==431].copy()
qs = np.quantile(tr.oof_med, np.linspace(0,1,26))
tr["b"] = pd.cut(tr.oof_med, qs, include_lowest=True)
lut = tr.groupby("b", observed=True).future_spend_4w.median()
te["b"] = pd.cut(te.oof_med, qs, include_lowest=True)
pc = te["b"].map(lut).astype(float).fillna(te.oof_med)
print("day431 raw MAE %.3f -> binned-median calib MAE %.3f"%(np.abs(te.resid).mean(), np.abs(te.future_spend_4w-pc).mean()))
# 3) blend of oof_med with slight weight on oof_sq / shrinkage of top
for w in [0,0.1,0.2,0.3]:
    p = (1-w)*oof.oof_med + w*oof.oof_sq
    print("blend w_sq=%.1f MAE %.3f"%(w, np.abs(p-oof.future_spend_4w).mean()))
# shrink top tail: p = med if med<=q else q + 0.8*(med-q)
for k in [0.8,0.9,1.0]:
    q = np.quantile(oof.oof_med, 0.95)
    p = np.where(oof.oof_med>q, q + k*(oof.oof_med-q), oof.oof_med)
    print("tail shrink k=%.1f MAE %.3f"%(k, np.abs(p-oof.future_spend_4w).mean()))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
for name in ["feats_v4","feats_v5","feats_v6"]:
    df = agent_api.load_saved(name+".parquet")
    print("==", name, df.shape)
    print([c for c in df.columns if c not in agent_api.load_saved("feats_v3.parquet").columns])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time

DEPTS = ["GROCER","DRUGG","PRODUC","COSMET","NUTRIT","MEAT","MEAT-P","DELI","PASTRY","FLORAL"]

def build(view, d):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    pr = view.table("products")[["product_id","department","brand"]]
    tx = tx.merge(pr, on="product_id", how="left")
    tx["dept"] = tx["department"].fillna("NA")
    tx["ispriv"] = (tx["brand"]=="Private").astype(np.float32)
    out = pd.DataFrame(index=hh)
    def agg(t, tag):
        g = t.groupby("household_key")
        f = pd.DataFrame({
            "spend": g["sales_value"].sum(),
            "baskets": g["basket_id"].nunique(),
            "prods": g["product_id"].nunique(),
            "stores": g["store_id"].nunique(),
            "qty": g["quantity"].sum(),
            "retail_disc": g["retail_disc"].sum(),
            "coupon_disc": g["coupon_disc"].sum()})
        mb = t.groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").max()
        f["spend_max_basket"] = mb
        f.columns = [f"{c}_{tag}" for c in f.columns]
        return f
    for w, tag in [(None,"all"),(7,"7"),(14,"14"),(28,"28"),(56,"56"),(84,"84"),(112,"112"),(224,"224")]:
        t = tx if w is None else tx[tx.day > d-w]
        out = out.join(agg(t, tag))
    # recency / cadence
    g = tx.groupby("household_key")
    out["days_since_last"] = d - g["day"].max()
    out["days_since_first"] = d - g["day"].min()
    age = np.maximum(tx["day"].values, 0)
    dec = (tx["sales_value"].values * np.exp(-np.log(2)*(d-tx["day"].values)/112.0))
    out["spend_decay_112"] = g.apply(lambda x: 0) # placeholder replaced below
    out.drop(columns=["spend_decay_112"], inplace=True)
    out["spend_decay_112"] = pd.Series(dec, index=tx.index).groupby(tx["household_key"]).sum().reindex(hh).fillna(0)
    out["trend_28_56"] = (out["spend_28"]+1)/(out["spend_56"]+1)
    out["trend_56_112"] = (out["spend_56"]+1)/(out["spend_112"]+1)
    out["trend_84_224"] = (out["spend_84"]+1)/(out["spend_224"]+1)
    out["share_recent_all"] = out["spend_28"]/(out["spend_all"]+1)
    out["avg_basket_84"] = out["spend_84"]/out["baskets_84"].replace(0,np.nan)
    out["avg_basket_28"] = out["spend_28"]/out["baskets_28"].replace(0,np.nan)
    t84 = tx[tx.day > d-84]
    bs = t84.groupby(["household_key","basket_id"])["sales_value"].sum()
    bstat = bs.groupby("household_key").agg(basket_std_84="std", basket_min_84="min")
    cnt = t84.groupby(["household_key","basket_id"]).size()
    out = out.join(bstat)
    out["items_per_basket_84"] = cnt.groupby("household_key").mean()
    out["qty_per_basket_84"] = t84.groupby(["household_key","basket_id"])["quantity"].sum().groupby("household_key").mean()
    out["baskets_per_day_84"] = out["baskets_84"]/84.0
    out["active_28"] = (out["baskets_28"]>0).astype(np.float32)
    out["active_84"] = (out["baskets_84"]>0).astype(np.float32)
    out["active_112"] = (out["baskets_112"]>0).astype(np.float32)
    bd = t84.groupby("household_key")["day"].apply(lambda s: np.sort(s.unique()))
    out["gap_mean_112"] = bd.apply(lambda a: np.mean(np.diff(a)) if len(a)>1 else 0)
    out["gap_std_112"] = bd.apply(lambda a: np.std(np.diff(a)) if len(a)>1 else 0)
    t84 = t84.assign(hour=(t84["trans_time"]//100))
    h = t84.groupby("household_key")["hour"]
    out["hour_mean_84"] = h.mean(); out["hour_std_84"] = h.std()
    out["evening_share_84"] = t84.assign(ev=(t84["hour"]>=17).astype(float)).groupby("household_key")["ev"].mean()
    # dept shares 84d
    t84d = t84.groupby(["household_key","dept"])["sales_value"].sum().unstack(fill_value=0.0)
    tot = t84d.sum(axis=1)
    for dep in DEPTS:
        col = t84d[dep] if dep in t84d.columns else pd.Series(0.0, index=t84d.index)
        out[f"dept_share_{dep}"] = (col/(tot+1e-9)).reindex(hh).fillna(0)
    other = t84d.drop(columns=[c for c in DEPTS if c in t84d.columns]).sum(axis=1)
    out["dept_other_share_84"] = (other/(tot+1e-9)).reindex(hh).fillna(0)
    out["n_depts_84"] = (t84d>0).sum(axis=1).reindex(hh).fillna(0)
    out["private_share_84"] = t84.groupby("household_key")["ispriv"].mean().reindex(hh).fillna(0)
    st = t84.groupby(["household_key","store_id"])["sales_value"].sum()
    out["top_store_share_84"] = (st.groupby("household_key").max()/(tot+1e-9)).reindex(hh).fillna(0)
    out["spend_rate_all_28"] = out["spend_all"]/(out["days_since_first"]+1)*28
    # campaigns / redemptions
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hh)]
    out["n_campaigns"] = ct.groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    for t_ in ["TypeA","TypeB","TypeC"]:
        out[f"ct_{t_}"] = ct[ct.description==t_].groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    rd = view.table("coupon_redemptions")
    rd = rd[rd.household_key.isin(hh)]
    out["redemp_all"] = rd.groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_84"] = rd[rd.day > d-84].groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_28"] = rd[rd.day > d-28].groupby("household_key").size().reindex(hh).fillna(0)
    out["days_since_redemp"] = (d - rd.groupby("household_key")["day"].max()).reindex(hh).fillna(9999)
    # display exposure
    w0 = (d+8)//7
    dm = view.table("display_mailer")
    dm = dm[(dm.week_no > w0-4) & (dm.week_no <= w0)]
    disp_ids = set(pd.to_numeric(dm["display"], errors="coerce").pipe(lambda s: s[s>0]).index.get_level_values(0)) if len(dm) else set()
    disp_ids = set(dm.loc[pd.to_numeric(dm["display"], errors="coerce").fillna(0)>0, "product_id"])
    t28 = tx[tx.day > d-28]
    sd = t28[t28.product_id.isin(disp_ids)].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    out["spend28_on_disp"] = sd
    out["share28_on_disp"] = sd/(out["spend_28"]+1)
    # demographics
    dem = view.table("demographics").set_index("household_key")
    m1 = {f"Age Group{i}":i for i in range(1,7)}
    m2 = {"X":0,"Y":1,"Z":2}
    m3 = {f"Level{i}":i for i in range(1,13)}
    m4 = {"1":1,"2":2,"3":3,"4":4,"5+":5}
    m5 = {f"Group{i}":i for i in range(1,7)}
    m6 = {"Renter":0,"Probable Renter":1,"Unknown":2,"Probable Owner":3,"Homeowner":4}
    m7 = {"None/Unknown":0,"1":1,"2":2,"3+":3}
    out["dem_classification_1"] = dem["classification_1"].map(m1).reindex(hh)
    out["dem_classification_2"] = dem["classification_2"].map(m2).reindex(hh)
    out["dem_classification_3"] = dem["classification_3"].map(m3).reindex(hh)
    out["dem_classification_4"] = dem["classification_4"].map(m4).reindex(hh)
    out["dem_classification_5"] = dem["classification_5"].map(m5).reindex(hh)
    out["dem_homeowner_desc"] = dem["homeowner_desc"].map(m6).reindex(hh)
    out["dem_kid_category_desc"] = dem["kid_category_desc"].map(m7).reindex(hh)
    out["has_demo"] = out["dem_classification_1"].notna().astype(np.float32)
    for c in [c for c in out.columns if c.startswith("dem_")]:
        out[c] = out[c].fillna(-1)
    out["week_of_year"] = ((w0-1) % 52) + 1
    # seasonal lags
    for lag in [84,112,168,252,308,364]:
        t = tx[(tx.day > d-lag-28) & (tx.day <= d-lag)]
        gg = t.groupby("household_key")
        out[f"lag{lag}_spend"] = gg["sales_value"].sum().reindex(hh).fillna(0)
        out[f"lag{lag}_bask"] = gg["basket_id"].nunique().reindex(hh).fillna(0)
    t = tx[(tx.day > d-364-56) & (tx.day <= d-364)]
    out["lag364_8w"] = t.groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    # NEW E014: seasonal interaction / trend features
    out["seas_ratio_364"] = out["lag364_spend"]/(out["spend_28"]+10)
    out["seas_ratio_308"] = out["lag308_spend"]/(out["spend_28"]+10)
    out["seas_trend"] = (out["lag364_spend"]+1)/(out["lag252_spend"]+1)
    out["seas_diff_364"] = out["lag364_spend"] - out["spend_28"]
    out["active_y1"] = (out["lag364_bask"]>0).astype(np.float32)
    out["seas_ratio_364_8w"] = out["lag364_8w"]/(out["spend_56"]+10)
    out["seas_avg_y1"] = (out["lag308_spend"]+out["lag252_spend"]+out["lag364_spend"])/3.0
    out["seas_vs_year_avg"] = out["lag364_spend"]/(out["seas_avg_y1"]+10)
    out = out.astype(np.float32)
    return out

t0=time.time()
feats = agent_api.build_features(build)
print("built", feats.shape, "%.1fs"%(time.time()-t0))
print(feats.snapshot_day.value_counts().sort_index())
print("NaN share:", feats.isna().mean().mean().round(4))
print(feats.head(3).T.head(40))
p = agent_api.save_table(feats, "feats_e014.parquet")
print("saved", p)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time

DEPTS = ["GROCER","DRUGG","PRODUC","COSMET","NUTRIT","MEAT","MEAT-P","DELI","PASTRY","FLORAL"]

def build(view, d):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    pr = view.table("products")[["product_id","department","brand"]]
    tx = tx.merge(pr, on="product_id", how="left")
    tx["dept"] = tx["department"].astype(str).replace("nan","NA")
    tx["ispriv"] = (tx["brand"].astype(str)=="Private").astype(np.float32)
    out = pd.DataFrame(index=hh)
    def agg(t, tag):
        g = t.groupby("household_key")
        f = pd.DataFrame({
            "spend": g["sales_value"].sum(),
            "baskets": g["basket_id"].nunique(),
            "prods": g["product_id"].nunique(),
            "stores": g["store_id"].nunique(),
            "qty": g["quantity"].sum(),
            "retail_disc": g["retail_disc"].sum(),
            "coupon_disc": g["coupon_disc"].sum()})
        f["spend_max_basket"] = t.groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").max()
        f.columns = [f"{c}_{tag}" for c in f.columns]
        return f
    for w, tag in [(None,"all"),(7,"7"),(14,"14"),(28,"28"),(56,"56"),(84,"84"),(112,"112"),(224,"224")]:
        t = tx if w is None else tx[tx.day > d-w]
        out = out.join(agg(t, tag))
    g = tx.groupby("household_key")
    out["days_since_last"] = d - g["day"].max()
    out["days_since_first"] = d - g["day"].min()
    dec = pd.Series(tx["sales_value"].values*np.exp(-np.log(2)*(d-tx["day"].values)/112.0), index=tx.index)
    out["spend_decay_112"] = dec.groupby(tx["household_key"]).sum().reindex(hh).fillna(0)
    out["trend_28_56"] = (out["spend_28"]+1)/(out["spend_56"]+1)
    out["trend_56_112"] = (out["spend_56"]+1)/(out["spend_112"]+1)
    out["trend_84_224"] = (out["spend_84"]+1)/(out["spend_224"]+1)
    out["share_recent_all"] = out["spend_28"]/(out["spend_all"]+1)
    out["avg_basket_84"] = out["spend_84"]/out["baskets_84"].replace(0,np.nan)
    out["avg_basket_28"] = out["spend_28"]/out["baskets_28"].replace(0,np.nan)
    t84 = tx[tx.day > d-84]
    bs = t84.groupby(["household_key","basket_id"])["sales_value"].sum()
    out = out.join(bs.groupby("household_key").agg(basket_std_84="std", basket_min_84="min"))
    cnt = t84.groupby(["household_key","basket_id"]).size()
    out["items_per_basket_84"] = cnt.groupby("household_key").mean()
    out["qty_per_basket_84"] = t84.groupby(["household_key","basket_id"])["quantity"].sum().groupby("household_key").mean()
    out["baskets_per_day_84"] = out["baskets_84"]/84.0
    out["active_28"] = (out["baskets_28"]>0).astype(np.float32)
    out["active_84"] = (out["baskets_84"]>0).astype(np.float32)
    out["active_112"] = (out["baskets_112"]>0).astype(np.float32)
    bd = t84.groupby("household_key")["day"].apply(lambda s: np.sort(s.unique()))
    out["gap_mean_112"] = bd.apply(lambda a: np.mean(np.diff(a)) if len(a)>1 else 0)
    out["gap_std_112"] = bd.apply(lambda a: np.std(np.diff(a)) if len(a)>1 else 0)
    t84 = t84.assign(hour=(t84["trans_time"]//100))
    h = t84.groupby("household_key")["hour"]
    out["hour_mean_84"] = h.mean(); out["hour_std_84"] = h.std()
    out["evening_share_84"] = t84.assign(ev=(t84["hour"]>=17).astype(float)).groupby("household_key")["ev"].mean()
    t84d = t84.groupby(["household_key","dept"])["sales_value"].sum().unstack(fill_value=0.0)
    tot = t84d.sum(axis=1)
    for dep in DEPTS:
        col = t84d[dep] if dep in t84d.columns else pd.Series(0.0, index=t84d.index)
        out[f"dept_share_{dep}"] = (col/(tot+1e-9)).reindex(hh).fillna(0)
    other = t84d.drop(columns=[c for c in DEPTS if c in t84d.columns]).sum(axis=1)
    out["dept_other_share_84"] = (other/(tot+1e-9)).reindex(hh).fillna(0)
    out["n_depts_84"] = (t84d>0).sum(axis=1).reindex(hh).fillna(0)
    out["private_share_84"] = t84.groupby("household_key")["ispriv"].mean().reindex(hh).fillna(0)
    st = t84.groupby(["household_key","store_id"])["sales_value"].sum()
    out["top_store_share_84"] = (st.groupby("household_key").max()/(tot+1e-9)).reindex(hh).fillna(0)
    out["spend_rate_all_28"] = out["spend_all"]/(out["days_since_first"]+1)*28
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hh)]
    out["n_campaigns"] = ct.groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    for t_ in ["TypeA","TypeB","TypeC"]:
        out[f"ct_{t_}"] = ct[ct.description==t_].groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    rd = view.table("coupon_redemptions")
    rd = rd[rd.household_key.isin(hh)]
    out["redemp_all"] = rd.groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_84"] = rd[rd.day > d-84].groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_28"] = rd[rd.day > d-28].groupby("household_key").size().reindex(hh).fillna(0)
    out["days_since_redemp"] = (d - rd.groupby("household_key")["day"].max()).reindex(hh).fillna(9999)
    w0 = (d+8)//7
    dm = view.table("display_mailer")
    dm = dm[(dm.week_no > w0-4) & (dm.week_no <= w0)]
    disp_ids = set(dm.loc[pd.to_numeric(dm["display"], errors="coerce").fillna(0)>0, "product_id"])
    t28 = tx[tx.day > d-28]
    sd = t28[t28.product_id.isin(disp_ids)].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    out["spend28_on_disp"] = sd
    out["share28_on_disp"] = sd/(out["spend_28"]+1)
    dem = view.table("demographics").set_index("household_key")
    m1 = {f"Age Group{i}":i for i in range(1,7)}
    m2 = {"X":0,"Y":1,"Z":2}
    m3 = {f"Level{i}":i for i in range(1,13)}
    m4 = {"1":1,"2":2,"3":3,"4":4,"5+":5}
    m5 = {f"Group{i}":i for i in range(1,7)}
    m6 = {"Renter":0,"Probable Renter":1,"Unknown":2,"Probable Owner":3,"Homeowner":4}
    m7 = {"None/Unknown":0,"1":1,"2":2,"3+":3}
    out["dem_classification_1"] = dem["classification_1"].map(m1).reindex(hh)
    out["dem_classification_2"] = dem["classification_2"].map(m2).reindex(hh)
    out["dem_classification_3"] = dem["classification_3"].map(m3).reindex(hh)
    out["dem_classification_4"] = dem["classification_4"].map(m4).reindex(hh)
    out["dem_classification_5"] = dem["classification_5"].map(m5).reindex(hh)
    out["dem_homeowner_desc"] = dem["homeowner_desc"].map(m6).reindex(hh)
    out["dem_kid_category_desc"] = dem["kid_category_desc"].map(m7).reindex(hh)
    out["has_demo"] = out["dem_classification_1"].notna().astype(np.float32)
    for c in [c for c in out.columns if c.startswith("dem_")]:
        out[c] = out[c].fillna(-1)
    out["week_of_year"] = ((w0-1) % 52) + 1
    for lag in [84,112,168,252,308,364]:
        t = tx[(tx.day > d-lag-28) & (tx.day <= d-lag)]
        gg = t.groupby("household_key")
        out[f"lag{lag}_spend"] = gg["sales_value"].sum().reindex(hh).fillna(0)
        out[f"lag{lag}_bask"] = gg["basket_id"].nunique().reindex(hh).fillna(0)
    t = tx[(tx.day > d-364-56) & (tx.day <= d-364)]
    out["lag364_8w"] = t.groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    out["seas_ratio_364"] = out["lag364_spend"]/(out["spend_28"]+10)
    out["seas_ratio_308"] = out["lag308_spend"]/(out["spend_28"]+10)
    out["seas_trend"] = (out["lag364_spend"]+1)/(out["lag252_spend"]+1)
    out["seas_diff_364"] = out["lag364_spend"] - out["spend_28"]
    out["active_y1"] = (out["lag364_bask"]>0).astype(np.float32)
    out["seas_ratio_364_8w"] = out["lag364_8w"]/(out["spend_56"]+10)
    out["seas_avg_y1"] = (out["lag308_spend"]+out["lag252_spend"]+out["lag364_spend"])/3.0
    out["seas_vs_year_avg"] = out["lag364_spend"]/(out["seas_avg_y1"]+10)
    out = out.astype(np.float32)
    return out

t0=time.time()
feats = agent_api.build_features(build)
print("built", feats.shape, "%.1fs"%(time.time()-t0))
print(feats.snapshot_day.value_counts().sort_index())
print("NaN share: %.5f"%feats.isna().mean().mean())
p = agent_api.save_table(feats, "feats_e014.parquet")
print("saved", p)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time

DEPTS = ["GROCER","DRUGG","PRODUC","COSMET","NUTRIT","MEAT","MEAT-P","DELI","PASTRY","FLORAL"]

def build(view, d):
    hh = pd.Index(view.households, name="household_key")
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    pr = view.table("products")[["product_id","department","brand"]]
    tx = tx.merge(pr, on="product_id", how="left")
    tx["dept"] = tx["department"].astype(str).replace("nan","NA")
    tx["ispriv"] = (tx["brand"].astype(str)=="Private").astype(np.float32)
    out = pd.DataFrame(index=hh)
    def agg(t, tag):
        g = t.groupby("household_key")
        f = pd.DataFrame({
            "spend": g["sales_value"].sum(),
            "baskets": g["basket_id"].nunique(),
            "prods": g["product_id"].nunique(),
            "stores": g["store_id"].nunique(),
            "qty": g["quantity"].sum(),
            "retail_disc": g["retail_disc"].sum(),
            "coupon_disc": g["coupon_disc"].sum()})
        f["spend_max_basket"] = t.groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").max()
        f.columns = [f"{c}_{tag}" for c in f.columns]
        return f
    for w, tag in [(None,"all"),(7,"7"),(14,"14"),(28,"28"),(56,"56"),(84,"84"),(112,"112"),(224,"224")]:
        t = tx if w is None else tx[tx.day > d-w]
        out = out.join(agg(t, tag))
    g = tx.groupby("household_key")
    out["days_since_last"] = d - g["day"].max()
    out["days_since_first"] = d - g["day"].min()
    dec = pd.Series(tx["sales_value"].values*np.exp(-np.log(2)*(d-tx["day"].values)/112.0), index=tx.index)
    out["spend_decay_112"] = dec.groupby(tx["household_key"]).sum().reindex(hh).fillna(0)
    out["trend_28_56"] = (out["spend_28"]+1)/(out["spend_56"]+1)
    out["trend_56_112"] = (out["spend_56"]+1)/(out["spend_112"]+1)
    out["trend_84_224"] = (out["spend_84"]+1)/(out["spend_224"]+1)
    out["share_recent_all"] = out["spend_28"]/(out["spend_all"]+1)
    out["avg_basket_84"] = out["spend_84"]/out["baskets_84"].replace(0,np.nan)
    out["avg_basket_28"] = out["spend_28"]/out["baskets_28"].replace(0,np.nan)
    t84 = tx[tx.day > d-84]
    bs = t84.groupby(["household_key","basket_id"])["sales_value"].sum()
    out = out.join(bs.groupby("household_key").agg(basket_std_84="std", basket_min_84="min"))
    cnt = t84.groupby(["household_key","basket_id"]).size()
    out["items_per_basket_84"] = cnt.groupby("household_key").mean()
    out["qty_per_basket_84"] = t84.groupby(["household_key","basket_id"])["quantity"].sum().groupby("household_key").mean()
    out["baskets_per_day_84"] = out["baskets_84"]/84.0
    out["active_28"] = (out["baskets_28"]>0).astype(np.float32)
    out["active_84"] = (out["baskets_84"]>0).astype(np.float32)
    out["active_112"] = (out["baskets_112"]>0).astype(np.float32)
    bd = t84.groupby("household_key")["day"].apply(lambda s: np.sort(s.unique()))
    out["gap_mean_112"] = bd.apply(lambda a: np.mean(np.diff(a)) if len(a)>1 else 0)
    out["gap_std_112"] = bd.apply(lambda a: np.std(np.diff(a)) if len(a)>1 else 0)
    t84 = t84.assign(hour=(t84["trans_time"]//100))
    h = t84.groupby("household_key")["hour"]
    out["hour_mean_84"] = h.mean(); out["hour_std_84"] = h.std()
    out["evening_share_84"] = t84.assign(ev=(t84["hour"]>=17).astype(float)).groupby("household_key")["ev"].mean()
    t84d = t84.groupby(["household_key","dept"])["sales_value"].sum().unstack(fill_value=0.0)
    tot = t84d.sum(axis=1)
    for dep in DEPTS:
        col = t84d[dep] if dep in t84d.columns else pd.Series(0.0, index=t84d.index)
        out[f"dept_share_{dep}"] = (col/(tot+1e-9)).reindex(hh).fillna(0)
    other = t84d.drop(columns=[c for c in DEPTS if c in t84d.columns]).sum(axis=1)
    out["dept_other_share_84"] = (other/(tot+1e-9)).reindex(hh).fillna(0)
    out["n_depts_84"] = (t84d>0).sum(axis=1).reindex(hh).fillna(0)
    out["private_share_84"] = t84.groupby("household_key")["ispriv"].mean().reindex(hh).fillna(0)
    st = t84.groupby(["household_key","store_id"])["sales_value"].sum()
    out["top_store_share_84"] = (st.groupby("household_key").max()/(tot+1e-9)).reindex(hh).fillna(0)
    out["spend_rate_all_28"] = out["spend_all"]/(out["days_since_first"]+1)*28
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hh)]
    out["n_campaigns"] = ct.groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    for t_ in ["TypeA","TypeB","TypeC"]:
        out[f"ct_{t_}"] = ct[ct.description==t_].groupby("household_key")["campaign"].nunique().reindex(hh).fillna(0)
    rd = view.table("coupon_redemptions")
    rd = rd[rd.household_key.isin(hh)]
    out["redemp_all"] = rd.groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_84"] = rd[rd.day > d-84].groupby("household_key").size().reindex(hh).fillna(0)
    out["redemp_28"] = rd[rd.day > d-28].groupby("household_key").size().reindex(hh).fillna(0)
    out["days_since_redemp"] = (d - rd.groupby("household_key")["day"].max()).reindex(hh).fillna(9999)
    w0 = (d+8)//7
    dm = view.table("display_mailer")
    dm = dm[(dm.week_no > w0-4) & (dm.week_no <= w0)]
    disp_ids = set(dm.loc[pd.to_numeric(dm["display"], errors="coerce").fillna(0)>0, "product_id"])
    t28 = tx[tx.day > d-28]
    sd = t28[t28.product_id.isin(disp_ids)].groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    out["spend28_on_disp"] = sd
    out["share28_on_disp"] = sd/(out["spend_28"]+1)
    dem = view.table("demographics").set_index("household_key")
    m1 = {f"Age Group{i}":i for i in range(1,7)}
    m2 = {"X":0,"Y":1,"Z":2}
    m3 = {f"Level{i}":i for i in range(1,13)}
    m4 = {"1":1,"2":2,"3":3,"4":4,"5+":5}
    m5 = {f"Group{i}":i for i in range(1,7)}
    m6 = {"Renter":0,"Probable Renter":1,"Unknown":2,"Probable Owner":3,"Homeowner":4}
    m7 = {"None/Unknown":0,"1":1,"2":2,"3+":3}
    for col, mp in [("classification_1",m1),("classification_2",m2),("classification_3",m3),
                    ("classification_4",m4),("classification_5",m5),("homeowner_desc",m6),
                    ("kid_category_desc",m7)]:
        out[f"dem_{col}"] = dem[col].astype(str).map(mp).reindex(hh).astype("float")
    out["has_demo"] = out["dem_classification_1"].notna().astype(np.float32)
    for c in [c for c in out.columns if c.startswith("dem_")]:
        out[c] = out[c].fillna(-1)
    out["week_of_year"] = ((w0-1) % 52) + 1
    for lag in [84,112,168,252,308,364]:
        t = tx[(tx.day > d-lag-28) & (tx.day <= d-lag)]
        gg = t.groupby("household_key")
        out[f"lag{lag}_spend"] = gg["sales_value"].sum().reindex(hh).fillna(0)
        out[f"lag{lag}_bask"] = gg["basket_id"].nunique().reindex(hh).fillna(0)
    t = tx[(tx.day > d-364-56) & (tx.day <= d-364)]
    out["lag364_8w"] = t.groupby("household_key")["sales_value"].sum().reindex(hh).fillna(0)
    out["seas_ratio_364"] = out["lag364_spend"]/(out["spend_28"]+10)
    out["seas_ratio_308"] = out["lag308_spend"]/(out["spend_28"]+10)
    out["seas_trend"] = (out["lag364_spend"]+1)/(out["lag252_spend"]+1)
    out["seas_diff_364"] = out["lag364_spend"] - out["spend_28"]
    out["active_y1"] = (out["lag364_bask"]>0).astype(np.float32)
    out["seas_ratio_364_8w"] = out["lag364_8w"]/(out["spend_56"]+10)
    out["seas_avg_y1"] = (out["lag308_spend"]+out["lag252_spend"]+out["lag364_spend"])/3.0
    out["seas_vs_year_avg"] = out["lag364_spend"]/(out["seas_avg_y1"]+10)
    out = out.astype(np.float32)
    return out

t0=time.time()
feats = agent_api.build_features(build)
print("built", feats.shape, "%.1fs"%(time.time()-t0))
print(feats.snapshot_day.value_counts().sort_index())
print("NaN share: %.5f"%feats.isna().mean().mean())
p = agent_api.save_table(feats, "feats_e014.parquet")
print("saved", p)


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

v3 = agent_api.load_saved("feats_v3.parquet")
seas = agent_api.load_saved("feats_seasonal.parquet").drop(columns=["snapshot_day"])
df = v3.merge(seas, on="household_key", suffixes=("","_s"))
# sanity: same snapshot_day
assert (df.snapshot_day == df.snapshot_day_s).all()
df = df.drop(columns=["snapshot_day_s"])
# new seasonal interaction features
df["seas_ratio_364"] = df["lag364_spend"]/(df["spend_28"]+10)
df["seas_ratio_308"] = df["lag308_spend"]/(df["spend_28"]+10)
df["seas_trend"] = (df["lag364_spend"]+1)/(df["lag252_spend"]+1)
df["seas_diff_364"] = df["lag364_spend"] - df["spend_28"]
df["active_y1"] = (df["lag364_bask"]>0).astype(np.float32)
df["seas_ratio_364_8w"] = df["lag364_8w"]/(df["spend_56"]+10)
df["seas_avg_y1"] = (df["lag308_spend"]+df["lag252_spend"]+df["lag364_spend"])/3.0
df["seas_vs_year_avg"] = df["lag364_spend"]/(df["seas_avg_y1"]+10)
feats_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(feats_cols))
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
tr = df[df.future_spend_4w.notna()].copy()
va = df[df.future_spend_4w.isna()].copy()
print("train rows", len(tr), "val rows", len(va))
Xtr, ytr = tr[feats_cols].values.astype(np.float32), tr.future_spend_4w.values
Xva = va[feats_cols].values.astype(np.float32)

t0=time.time()
preds = []
def add(p, w=1.0):
    preds.append((p, w))
# XGB median x2 seeds
for s in [7, 17]:
    m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:quantileerror",
                     quantile_alpha=0.5, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
    m.fit(Xtr, ytr); add(m.predict(Xva))
print("xgb med done %.0fs"%(time.time()-t0))
# XGB squared x1
m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                 tree_method="hist", n_jobs=8, random_state=7, verbosity=0)
m.fit(Xtr, ytr); add(m.predict(Xva))
print("xgb sq done %.0fs"%(time.time()-t0))
# HGB quantile x2 seeds
for s in [7, 17]:
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=s)
    m.fit(Xtr, ytr); add(m.predict(Xva))
print("hgb done %.0fs"%(time.time()-t0))
W = np.array([w for _,w in preds], float)
P = np.vstack([p for p,_ in preds])
pred = (W[:,None]*P).sum(0)/W.sum()
out = va[["household_key","snapshot_day"]].copy()
out["prediction"] = pred
print(out.prediction.describe().round(2))
p = agent_api.save_table(out, "pred_e014.parquet")
print("saved", p, "%.0fs"%(time.time()-t0))


# ---- cell ----
import agent_api, pandas as pd
v3 = agent_api.load_saved("feats_v3.parquet")
seas = agent_api.load_saved("feats_seasonal.parquet")
print(v3.shape, seas.shape, seas.columns[:3].tolist())
df = v3.merge(seas.drop(columns=["snapshot_day"]), on="household_key")
print(df.shape)
print("match:", (df.snapshot_day_x == df.snapshot_day_y).all() if "snapshot_day_y" in df.columns else "n/a")
print([c for c in df.columns if "snapshot" in c])


# ---- cell ----
import agent_api, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
from xgboost import XGBRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

v3 = agent_api.load_saved("feats_v3.parquet")
seas = agent_api.load_saved("feats_seasonal.parquet")
df = v3.merge(seas, on=["household_key","snapshot_day"])
print(df.shape)
df["seas_ratio_364"] = df["lag364_spend"]/(df["spend_28"]+10)
df["seas_ratio_308"] = df["lag308_spend"]/(df["spend_28"]+10)
df["seas_trend"] = (df["lag364_spend"]+1)/(df["lag252_spend"]+1)
df["seas_diff_364"] = df["lag364_spend"] - df["spend_28"]
df["active_y1"] = (df["lag364_bask"]>0).astype(np.float32)
df["seas_ratio_364_8w"] = df["lag364_8w"]/(df["spend_56"]+10)
df["seas_avg_y1"] = (df["lag308_spend"]+df["lag252_spend"]+df["lag364_spend"])/3.0
df["seas_vs_year_avg"] = df["lag364_spend"]/(df["seas_avg_y1"]+10)
feats_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
print("n features:", len(feats_cols))
tt = agent_api.train_targets()
df = df.merge(tt, on=["household_key","snapshot_day"], how="left")
tr = df[df.future_spend_4w.notna()].copy()
va = df[df.future_spend_4w.isna()].copy()
print("train rows", len(tr), "val rows", len(va))
Xtr, ytr = tr[feats_cols].values.astype(np.float32), tr.future_spend_4w.values
Xva = va[feats_cols].values.astype(np.float32)
t0=time.time()
preds = []
for s in [7, 17]:
    m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:quantileerror",
                     quantile_alpha=0.5, tree_method="hist", n_jobs=8, random_state=s, verbosity=0)
    m.fit(Xtr, ytr); preds.append(m.predict(Xva))
print("xgb med done %.0fs"%(time.time()-t0))
m = XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7, min_child_weight=10,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
                 tree_method="hist", n_jobs=8, random_state=7, verbosity=0)
m.fit(Xtr, ytr); preds.append(m.predict(Xva))
print("xgb sq done %.0fs"%(time.time()-t0))
for s in [7, 17]:
    m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=400, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=s)
    m.fit(Xtr, ytr); preds.append(m.predict(Xva))
print("hgb done %.0fs"%(time.time()-t0))
P = np.vstack(preds)
pred = P.mean(0)
out = va[["household_key","snapshot_day"]].copy()
out["prediction"] = pred
print(out.prediction.describe().round(2))
p = agent_api.save_table(out, "pred_e014.parquet")
print("saved", p, "%.0fs"%(time.time()-t0))
