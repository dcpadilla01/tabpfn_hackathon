import pandas as pd, numpy as np, agent_api

def build(view, day):
    hh = view.households
    tx = view.table("transactions")
    idx = pd.Index(hh, name="household_key")
    feats = pd.DataFrame(index=idx)
    g = tx.groupby("household_key")
    def sp(d): return tx[tx.day > day-d].groupby("household_key").sales_value.sum()
    for d in (7,28,56,84,168):
        feats[f"spend_{d}"] = sp(d).reindex(idx).fillna(0)
    feats["spend_all"] = g.sales_value.sum().reindex(idx).fillna(0)
    feats["spend_prev28"] = (feats["spend_56"]-feats["spend_28"])
    feats["spend_prev56"] = (feats["spend_112"] if False else tx[(tx.day>day-112)&(tx.day<=day-56)].groupby("household_key").sales_value.sum()).reindex(idx).fillna(0)
    feats["trips_28"] = tx[tx.day>day-28].groupby("household_key").basket_id.nunique().reindex(idx).fillna(0)
    feats["trips_84"] = tx[tx.day>day-84].groupby("household_key").basket_id.nunique().reindex(idx).fillna(0)
    feats["trips_prev28"] = tx[(tx.day>day-56)&(tx.day<=day-28)].groupby("household_key").basket_id.nunique().reindex(idx).fillna(0)
    b84 = tx[tx.day>day-84].groupby(["household_key","basket_id"]).sales_value.sum()
    feats["basket_mean_84"] = b84.groupby("household_key").mean().reindex(idx).fillna(0)
    feats["basket_std_84"] = b84.groupby("household_key").std().reindex(idx).fillna(0)
    feats["basket_max_84"] = b84.groupby("household_key").max().reindex(idx).fillna(0)
    items = tx[tx.day>day-84].groupby(["household_key","basket_id"]).size()
    feats["items_per_basket_84"] = items.groupby("household_key").mean().reindex(idx).fillna(0)
    days = g.day.nunique()
    feats["median_gap_84"] = tx[tx.day>day-84].groupby("household_key").day.apply(lambda s: np.diff(np.sort(s.unique())).mean() if s.nunique()>1 else 28).reindex(idx).fillna(28)
    wk = tx[tx.day>day-84].copy(); wk["w"]=wk.day//7
    feats["active_weeks_84"] = wk.groupby("household_key").w.nunique().reindex(idx).fillna(0)
    feats["n_products_84"] = tx[tx.day>day-84].groupby("household_key").product_id.nunique().reindex(idx).fillna(0)
    feats["n_stores_84"] = tx[tx.day>day-84].groupby("household_key").store_id.nunique().reindex(idx).fillna(0)
    feats["coupon_disc_28"] = tx[tx.day>day-28].groupby("household_key").coupon_disc.sum().abs().reindex(idx).fillna(0)
    feats["retail_disc_84"] = tx[tx.day>day-84].groupby("household_key").retail_disc.sum().abs().reindex(idx).fillna(0)
    feats["qty_28"] = tx[tx.day>day-28].groupby("household_key").quantity.sum().reindex(idx).fillna(0)
    last = g.day.max()
    feats["recency"] = (day-last).reindex(idx).fillna(999)
    first = g.day.min()
    feats["tenure"] = (day-first).reindex(idx).fillna(0)
    feats["avg_basket_28"] = feats["spend_28"]/feats["trips_28"].replace(0,np.nan)
    feats["weekly_rate_84"] = feats["spend_84"]/12
    feats["ratio_28_prev"] = feats["spend_28"]/(feats["spend_prev28"]+1)
    feats["trend_28_84"] = feats["spend_28"]*3/(feats["spend_84"]+1)
    feats["spend_28_log"] = np.log1p(feats["spend_28"])
    feats["spend_84_log"] = np.log1p(feats["spend_84"])
    # NEW: campaign targeting
    try:
        camps = view.table("campaigns"); tg = view.table("campaign_targets")
        tgc = tg.merge(camps[["campaign","description","start_day","end_day"]], on="campaign", suffixes=("","_c"))
        tgc["desc"] = tgc["description_c"].fillna(tgc["description"])
        ever = tgc[tgc.start_day<=day].groupby("household_key").size()
        feats["n_camp_ever"] = ever.reindex(idx).fillna(0)
        r28 = tgc[(tgc.start_day>day-28)&(tgc.start_day<=day)].groupby("household_key").size()
        feats["n_camp_start28"] = r28.reindex(idx).fillna(0)
        act = tgc[(tgc.start_day<=day)&(tgc.end_day>=day)].groupby("household_key").size()
        feats["n_camp_active"] = act.reindex(idx).fillna(0)
        for t in ("TypeA","TypeB","TypeC"):
            tc = tgc[(tgc.start_day<=day)&(tgc.desc==t)].groupby("household_key").size()
            feats[f"camp_{t}"] = tc.reindex(idx).fillna(0)
        lastc = tgc[tgc.start_day<=day].groupby("household_key").start_day.max()
        feats["days_since_camp"] = (day-lastc).reindex(idx).fillna(999)
    except Exception as e:
        print("camp err", e)
    # NEW: coupon redemptions
    try:
        red = view.table("coupon_redemptions")
        feats["red_28"] = red[red.day>day-28].groupby("household_key").size().reindex(idx).fillna(0)
        feats["red_84"] = red[red.day>day-84].groupby("household_key").size().reindex(idx).fillna(0)
        feats["red_ever"] = red.groupby("household_key").size().reindex(idx).fillna(0)
        lastr = red.groupby("household_key").day.max()
        feats["days_since_red"] = (day-lastr).reindex(idx).fillna(999)
    except Exception as e:
        print("red err", e)
    # NEW: demographics (ordinal-encoded)
    try:
        d = view.table("demographics").set_index("household_key")
        feats["has_demo"] = d.index.isin(idx).astype(float).reindex(idx).fillna(0) if False else idx.isin(d.index).astype(float)
        def ordmap(s, pref, n):
            return s.map(lambda v: int(str(v).replace(pref,"")) if str(v).startswith(pref) else np.nan)
        feats["age_ord"] = ordmap(d.classification_1,"Age Group",6).reindex(idx)
        feats["lvl_ord"] = ordmap(d.classification_3,"Level",12).reindex(idx)
        feats["size_ord"] = d.classification_4.map(lambda v: 5 if str(v)=="5+" else (float(v) if str(v).isdigit() else np.nan)).reindex(idx)
        feats["grp5_ord"] = ordmap(d.classification_5,"Group",6).reindex(idx)
        feats["home_ord"] = d.homeowner_desc.map({"Homeowner":4,"Probable Owner":3,"Probable Renter":2,"Renter":1,"Unknown":0}).reindex(idx)
        feats["kids_ord"] = d.kid_category_desc.map({"None/Unknown":0,"1":1,"2":2,"3+":3}).reindex(idx)
        feats["cls2"] = d.classification_2.astype("category").cat.codes.replace(-1,np.nan).reindex(idx)
    except Exception as e:
        print("demo err", e)
    return feats

feats = agent_api.build_features(build)
print(feats.shape)
print(feats.isna().mean().sort_values(ascending=False).head(8))
t = agent_api.train_targets()
tr = feats.merge(t, on=["household_key","snapshot_day"])
print("train rows", len(tr))
agent_api.save_table(feats, "feats_v2.parquet")

import xgboost as xgb
drop = ["household_key","snapshot_day","future_spend_4w"]
X = tr.drop(columns=drop); y = tr.future_spend_4w
val = tr[tr.snapshot_day==431]; trn = tr[tr.snapshot_day!=431]
m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=5, n_jobs=4)
m.fit(trn.drop(columns=drop), trn.future_spend_4w, eval_set=[(val.drop(columns=drop), val.future_spend_4w)], verbose=False)
from sklearn.metrics import mean_absolute_error
p = m.predict(val.drop(columns=drop))
print("holdout 431 MAE", mean_absolute_error(val.future_spend_4w, p))
va = feats[feats.snapshot_day>=459]
va = va.copy(); va["prediction"] = m.predict(va.drop(columns=drop))
out = va[["household_key","snapshot_day","prediction"]]
agent_api.save_table(out, "pred_e002.parquet")
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(15))
