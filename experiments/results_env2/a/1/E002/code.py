print(agent_api.snapshot_days())
print(agent_api.describe_tables())
b = agent_api.baseline_features()
print(b.shape)
print(b.columns.tolist())
print(b.head())
t = agent_api.train_targets()
print(t.shape, t.future_spend_4w.describe())


# ---- cell ----
v = agent_api.snapshot(95)
print(type(v.households))
print(v.households if hasattr(v.households,'__shape__') else v.households.head())
print("day", v.day, "week", v.week)
tx = v.table("transactions")
print(tx.shape, tx.day.max())
ct = v.table("campaign_targets")
print(ct.shape, ct.head(3))
cr = v.table("coupon_redemptions")
print(cr.shape, cr.head(3))
dm = v.table("display_mailer")
print(dm.shape, dm.head(3))
prod = agent_api.snapshot().table("products")
print(prod.department.value_counts().head(15))


# ---- cell ----
def probe(view, day):
    print("DAY", day, "households", type(view.households), getattr(view.households,'shape',None))
    if view.households is not None:
        print(view.households.head(3))
    tx = view.table("transactions")
    print("tx", tx.shape, tx.day.min(), tx.day.max())
    ct = view.table("campaign_targets")
    print("ct", ct.shape, ct.head(2))
    cr = view.table("coupon_redemptions")
    print("cr", cr.shape)
    dm = view.table("display_mailer")
    print("dm", dm.shape, dm.week_no.max())
    print("week", view.week)
    return view.households.iloc[:2].to_frame("hh") if view.households is not None else None

f = agent_api.build_features(probe)
print(f.shape)


# ---- cell ----
def probe(view, day):
    print("DAY", day, "n_hh", len(view.households))
    tx = view.table("transactions")
    print("tx", tx.shape, tx.day.min(), tx.day.max())
    ct = view.table("campaign_targets")
    print("ct", ct.shape, ct.description.value_counts().to_dict())
    cr = view.table("coupon_redemptions")
    print("cr", cr.shape)
    dm = view.table("display_mailer")
    print("dm", dm.shape, dm.week_no.max())
    print("week", view.week)
    return pd.DataFrame(index=view.households[:2])

f = agent_api.build_features(probe)
print(f.shape)


# ---- cell ----
import xgboost as xgb, sklearn
print("xgb", xgb.__version__, "sklearn", sklearn.__version__)
prod = agent_api.snapshot().table("products")
print(prod.department.value_counts().head(20))
t = agent_api.train_targets()
print("zero rate", (t.future_spend_4w==0).mean())
print(t.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]).round(1))
tx = agent_api.snapshot(431).table("transactions")
print(tx[['coupon_disc','retail_disc','coupon_match_disc']].describe().round(2))
print("brands", prod.brand.value_counts(dropna=False).head())


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

TOP_DEPTS = ["GROCERY","DRUG GM","PRODUCE","MEAT","MEAT-PCKGD","DELI","PASTRY","COSMETICS","NUTRITION"]

def feats(view, day):
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    # --- trailing windows ---
    for w in (28,56,84,112,168,364):
        m = tx.day > day - w
        g = tx.loc[m].groupby("household_key")
        out[f"spend_{w}"] = g.sales_value.sum()
        out[f"bask_{w}"]  = g.basket_id.nunique()
        out[f"actdays_{w}"] = g.day.nunique()
        if w in (84,364):
            out[f"qty_{w}"] = g.quantity.sum()
            out[f"retail_disc_{w}"] = -g.retail_disc.sum()
            out[f"coupon_disc_{w}"] = -g.coupon_disc.sum()
    g = tx.groupby("household_key")
    out["recency"] = day - g.day.max()
    out["tenure"] = day - g.day.min()
    out["spend_total"] = g.sales_value.sum()
    out["bask_total"] = g.basket_id.nunique()
    # lag: same 4-week window one year earlier
    m = (tx.day > day-364) & (tx.day <= day-336)
    out["spend_lag1y"] = tx.loc[m].groupby("household_key").sales_value.sum()
    # trends / ratios
    eps = 1.0
    out["trend_28_84"] = out.spend_28 / (out.spend_84/3 + eps)
    out["trend_56_112"] = out.spend_56 / (out.spend_112/2 + eps)
    out["spend28_vs_364"] = out.spend_28 / (out.spend_364/13 + eps)
    out["avg_basket_84"] = out.spend_84 / (out.bask_84 + 0.5)
    out["items_basket_84"] = out.qty_84 / (out.bask_84 + 0.5)
    out["ipi"] = out.tenure / (out.bask_total + 1.0)
    out["zero28"] = (out.spend_28 <= 0).astype(float)
    # --- departments (last 84d) ---
    prod = view.table("products")[["product_id","department","brand"]]
    t84 = tx.loc[tx.day > day-84].merge(prod, on="product_id", how="left")
    pv = t84.pivot_table(index="household_key", columns="department", values="sales_value", aggfunc="sum")
    have = set(pv.columns)
    for d in TOP_DEPTS:
        col = pv[d] if d in have else pd.Series(0.0, index=hh)
        out[f"dep_{d[:6]}"] = col.reindex(hh).fillna(0.0)
    known = pd.Series(0.0, index=hh)
    for d in TOP_DEPTS:
        if d in have: known = known.add(pv[d].reindex(hh).fillna(0.0), fill_value=0)
    out["dep_other"] = out.spend_84 - known
    t364 = tx.loc[tx.day > day-364].merge(prod[["product_id","department"]], on="product_id", how="left")
    out["ndeps_364"] = t364.groupby("household_key").department.nunique().reindex(hh).fillna(0)
    pb = t84.brand.eq("Private").groupby(t84.household_key).mean().reindex(hh)
    out["private_share_84"] = pb.fillna(0.0)
    # --- time of day ---
    tt = tx.loc[tx.day > day-84, ["household_key","trans_time","sales_value"]]
    hr = tt.trans_time.fillna(0).astype(int) // 100
    out["eve_spend_share_84"] = tt.sales_value.where(hr>=17, 0).groupby(tt.household_key).sum().reindex(hh).fillna(0) / (out.spend_84+eps)
    # --- marketing exposure ---
    ct = view.table("campaign_targets")
    if len(ct):
        ctd = ct.drop_duplicates(["household_key","description"])
        pv2 = ctd.pivot_table(index="household_key", columns="description", values="campaign", aggfunc="count")
        for c in ["TypeA","TypeB","TypeC"]:
            out[f"camp_{c}"] = (pv2[c] if c in pv2.columns else pd.Series(dtype=float)).reindex(hh).fillna(0)
        out["camp_any"] = ct.groupby("household_key").campaign.nunique().reindex(hh).fillna(0)
    else:
        for c in ["TypeA","TypeB","TypeC"]: out[f"camp_{c}"] = 0.0
        out["camp_any"] = 0.0
    cr = view.table("coupon_redemptions")
    if len(cr):
        cr = cr[cr.household_key.isin(hh)]
        out["redemp_total"] = cr.groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_84"] = cr.loc[cr.day > day-84].groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_recency"] = (day - cr.groupby("household_key").day.max()).reindex(hh).fillna(999)
    else:
        out["redemp_total"] = 0.0; out["redemp_84"] = 0.0; out["redemp_recency"] = 999.0
    # --- demographics ---
    try:
        dm = view.table("demographics").set_index("household_key")
    except Exception:
        dm = None
    if dm is not None and len(dm):
        def num(s, pat):
            return s.str.extract(pat).astype(float)
        out["age_code"] = num(dm.classification_1, r"(\d+)").reindex(hh)
        out["c2_code"] = dm.classification_2.map({"X":0,"Y":1,"Z":2}).reindex(hh)
        out["income_code"] = num(dm.classification_3, r"(\d+)").reindex(hh)
        out["hhsize_code"] = dm.classification_4.replace({"5+":5}).astype(float).reindex(hh)
        out["c5_code"] = num(dm.classification_5, r"(\d+)").reindex(hh)
        out["homeown_code"] = dm.homeowner_desc.map({"Homeowner":4,"Probable Owner":3,"Probable Renter":2,"Renter":1,"Unknown":0}).reindex(hh)
        out["kids_code"] = dm.kid_category_desc.map({"None/Unknown":0,"1":1,"2":2,"3+":3}).reindex(hh)
        out["has_demo"] = dm.index.isin(hh).astype(float).reindex(hh).fillna(0)
    # --- snapshot calendar ---
    out["snapshot_day"] = float(day)
    wk = (day + 8) // 7
    out["week_of_year"] = float(wk)
    out["week_sin"] = np.sin(2*np.pi*wk/52); out["week_cos"] = np.cos(2*np.pi*wk/52)
    return out

t0 = time.time()
f = agent_api.build_features(feats)
print("built", f.shape, f"({time.time()-t0:.0f}s)")
print(f.isna().mean().sort_values(ascending=False).head(8))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

TOP_DEPTS = ["GROCERY","DRUG GM","PRODUCE","MEAT","MEAT-PCKGD","DELI","PASTRY","COSMETICS","NUTRITION"]

def feats(view, day):
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    for w in (28,56,84,112,168,364):
        m = tx.day > day - w
        g = tx.loc[m].groupby("household_key")
        out[f"spend_{w}"] = g.sales_value.sum()
        out[f"bask_{w}"]  = g.basket_id.nunique()
        out[f"actdays_{w}"] = g.day.nunique()
        if w in (84,364):
            out[f"qty_{w}"] = g.quantity.sum()
            out[f"retail_disc_{w}"] = -g.retail_disc.sum()
            out[f"coupon_disc_{w}"] = -g.coupon_disc.sum()
    g = tx.groupby("household_key")
    out["recency"] = day - g.day.max()
    out["tenure"] = day - g.day.min()
    out["spend_total"] = g.sales_value.sum()
    out["bask_total"] = g.basket_id.nunique()
    m = (tx.day > day-364) & (tx.day <= day-336)
    out["spend_lag1y"] = tx.loc[m].groupby("household_key").sales_value.sum()
    eps = 1.0
    out["trend_28_84"] = out.spend_28 / (out.spend_84/3 + eps)
    out["trend_56_112"] = out.spend_56 / (out.spend_112/2 + eps)
    out["spend28_vs_364"] = out.spend_28 / (out.spend_364/13 + eps)
    out["avg_basket_84"] = out.spend_84 / (out.bask_84 + 0.5)
    out["items_basket_84"] = out.qty_84 / (out.bask_84 + 0.5)
    out["ipi"] = out.tenure / (out.bask_total + 1.0)
    out["zero28"] = (out.spend_28 <= 0).astype(float)
    prod = view.table("products")[["product_id","department","brand"]]
    t84 = tx.loc[tx.day > day-84].merge(prod, on="product_id", how="left")
    pv = t84.pivot_table(index="household_key", columns="department", values="sales_value", aggfunc="sum")
    have = set(pv.columns)
    known = pd.Series(0.0, index=hh)
    for d in TOP_DEPTS:
        col = pv[d].reindex(hh).fillna(0.0) if d in have else pd.Series(0.0, index=hh)
        out[f"dep_{d[:6]}"] = col
        if d in have: known = known.add(col)
    out["dep_other"] = out.spend_84 - known
    t364 = tx.loc[tx.day > day-364].merge(prod[["product_id","department"]], on="product_id", how="left")
    out["ndeps_364"] = t364.groupby("household_key").department.nunique().reindex(hh).fillna(0)
    out["private_share_84"] = t84.brand.eq("Private").groupby(t84.household_key).mean().reindex(hh).fillna(0.0)
    tt = tx.loc[tx.day > day-84, ["household_key","trans_time","sales_value"]]
    hr = tt.trans_time.fillna(0).astype(int) // 100
    out["eve_spend_share_84"] = tt.sales_value.where(hr>=17, 0).groupby(tt.household_key).sum().reindex(hh).fillna(0) / (out.spend_84+eps)
    ct = view.table("campaign_targets")
    if len(ct):
        ctd = ct.drop_duplicates(["household_key","description"])
        pv2 = ctd.pivot_table(index="household_key", columns="description", values="campaign", aggfunc="count")
        for c in ["TypeA","TypeB","TypeC"]:
            out[f"camp_{c}"] = (pv2[c] if c in pv2.columns else pd.Series(dtype=float)).reindex(hh).fillna(0)
        out["camp_any"] = ct.groupby("household_key").campaign.nunique().reindex(hh).fillna(0)
    else:
        for c in ["TypeA","TypeB","TypeC"]: out[f"camp_{c}"] = 0.0
        out["camp_any"] = 0.0
    cr = view.table("coupon_redemptions")
    if len(cr):
        cr = cr[cr.household_key.isin(hh)]
        out["redemp_total"] = cr.groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_84"] = cr.loc[cr.day > day-84].groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_recency"] = (day - cr.groupby("household_key").day.max()).reindex(hh).fillna(999)
    else:
        out["redemp_total"] = 0.0; out["redemp_84"] = 0.0; out["redemp_recency"] = 999.0
    try:
        dm = view.table("demographics").set_index("household_key")
    except Exception:
        dm = None
    if dm is not None and len(dm):
        out["age_code"] = dm.classification_1.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["c2_code"] = dm.classification_2.map({"X":0,"Y":1,"Z":2}).reindex(hh)
        out["income_code"] = dm.classification_3.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["hhsize_code"] = dm.classification_4.replace({"5+":5}).astype(float).reindex(hh)
        out["c5_code"] = dm.classification_5.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["homeown_code"] = dm.homeowner_desc.map({"Homeowner":4,"Probable Owner":3,"Probable Renter":2,"Renter":1,"Unknown":0}).reindex(hh)
        out["kids_code"] = dm.kid_category_desc.map({"None/Unknown":0,"1":1,"2":2,"3+":3}).reindex(hh)
        out["has_demo"] = pd.Series(dm.index.isin(hh).astype(float), index=dm.index).reindex(hh).fillna(0)
    out["snapshot_day"] = float(day)
    wk = (day + 8) // 7
    out["week_of_year"] = float(wk)
    out["week_sin"] = np.sin(2*np.pi*wk/52); out["week_cos"] = np.cos(2*np.pi*wk/52)
    return out

t0 = time.time()
f = agent_api.build_features(feats)
print("built", f.shape, f"({time.time()-t0:.0f}s)")
print(f.isna().mean().sort_values(ascending=False).head(8))


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb

f = agent_api.load_saved("e002_features.parquet")
t = agent_api.train_targets()
d = f.merge(t, on=["household_key","snapshot_day"])
feats = [c for c in f.columns if c not in ("household_key","snapshot_day")]
d[feats] = d[feats].astype(float)
val = f[f.snapshot_day>=459].copy()
val[feats] = val[feats].astype(float)

params = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              objective="reg:squarederror", tree_method="hist", n_jobs=4)
model = xgb.XGBRegressor(**params)
model.fit(d[feats], d.future_spend_4w, eval_set=[(val[feats], val.future_spend_4w if 'future_spend_4w' in val else np.zeros(len(val)))], verbose=False)


# ---- cell ----
import numpy as np, pandas as pd, xgboost as xgb, time

TOP_DEPTS = ["GROCERY","DRUG GM","PRODUCE","MEAT","MEAT-PCKGD","DELI","PASTRY","COSMETICS","NUTRITION"]

def feats(view, day):
    hh = pd.Index(view.households, name="household_key")
    out = pd.DataFrame(index=hh)
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    for w in (28,56,84,112,168,364):
        m = tx.day > day - w
        g = tx.loc[m].groupby("household_key")
        out[f"spend_{w}"] = g.sales_value.sum()
        out[f"bask_{w}"]  = g.basket_id.nunique()
        out[f"actdays_{w}"] = g.day.nunique()
        if w in (84,364):
            out[f"qty_{w}"] = g.quantity.sum()
            out[f"retail_disc_{w}"] = -g.retail_disc.sum()
            out[f"coupon_disc_{w}"] = -g.coupon_disc.sum()
    g = tx.groupby("household_key")
    out["recency"] = day - g.day.max()
    out["tenure"] = day - g.day.min()
    out["spend_total"] = g.sales_value.sum()
    out["bask_total"] = g.basket_id.nunique()
    m = (tx.day > day-364) & (tx.day <= day-336)
    out["spend_lag1y"] = tx.loc[m].groupby("household_key").sales_value.sum()
    eps = 1.0
    out["trend_28_84"] = out.spend_28 / (out.spend_84/3 + eps)
    out["trend_56_112"] = out.spend_56 / (out.spend_112/2 + eps)
    out["spend28_vs_364"] = out.spend_28 / (out.spend_364/13 + eps)
    out["avg_basket_84"] = out.spend_84 / (out.bask_84 + 0.5)
    out["items_basket_84"] = out.qty_84 / (out.bask_84 + 0.5)
    out["ipi"] = out.tenure / (out.bask_total + 1.0)
    out["zero28"] = (out.spend_28 <= 0).astype(float)
    prod = view.table("products")[["product_id","department","brand"]]
    t84 = tx.loc[tx.day > day-84].merge(prod, on="product_id", how="left")
    pv = t84.pivot_table(index="household_key", columns="department", values="sales_value", aggfunc="sum")
    have = set(pv.columns)
    known = pd.Series(0.0, index=hh)
    for d in TOP_DEPTS:
        col = pv[d].reindex(hh).fillna(0.0) if d in have else pd.Series(0.0, index=hh)
        out[f"dep_{d[:6]}"] = col
        if d in have: known = known.add(col)
    out["dep_other"] = out.spend_84 - known
    t364 = tx.loc[tx.day > day-364].merge(prod[["product_id","department"]], on="product_id", how="left")
    out["ndeps_364"] = t364.groupby("household_key").department.nunique().reindex(hh).fillna(0)
    out["private_share_84"] = t84.brand.eq("Private").groupby(t84.household_key).mean().reindex(hh).fillna(0.0)
    tt = tx.loc[tx.day > day-84, ["household_key","trans_time","sales_value"]]
    hr = tt.trans_time.fillna(0).astype(int) // 100
    out["eve_spend_share_84"] = tt.sales_value.where(hr>=17, 0).groupby(tt.household_key).sum().reindex(hh).fillna(0) / (out.spend_84+eps)
    ct = view.table("campaign_targets")
    if len(ct):
        ctd = ct.drop_duplicates(["household_key","description"])
        pv2 = ctd.pivot_table(index="household_key", columns="description", values="campaign", aggfunc="count")
        for c in ["TypeA","TypeB","TypeC"]:
            out[f"camp_{c}"] = (pv2[c] if c in pv2.columns else pd.Series(dtype=float)).reindex(hh).fillna(0)
        out["camp_any"] = ct.groupby("household_key").campaign.nunique().reindex(hh).fillna(0)
    else:
        for c in ["TypeA","TypeB","TypeC"]: out[f"camp_{c}"] = 0.0
        out["camp_any"] = 0.0
    cr = view.table("coupon_redemptions")
    if len(cr):
        cr = cr[cr.household_key.isin(hh)]
        out["redemp_total"] = cr.groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_84"] = cr.loc[cr.day > day-84].groupby("household_key").day.count().reindex(hh).fillna(0)
        out["redemp_recency"] = (day - cr.groupby("household_key").day.max()).reindex(hh).fillna(999)
    else:
        out["redemp_total"] = 0.0; out["redemp_84"] = 0.0; out["redemp_recency"] = 999.0
    try:
        dm = view.table("demographics").set_index("household_key")
    except Exception:
        dm = None
    if dm is not None and len(dm):
        out["age_code"] = dm.classification_1.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["c2_code"] = dm.classification_2.map({"X":0,"Y":1,"Z":2}).reindex(hh)
        out["income_code"] = dm.classification_3.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["hhsize_code"] = dm.classification_4.replace({"5+":5}).astype(float).reindex(hh)
        out["c5_code"] = dm.classification_5.str.extract(r"(\d+)").astype(float).iloc[:,0].reindex(hh)
        out["homeown_code"] = dm.homeowner_desc.map({"Homeowner":4,"Probable Owner":3,"Probable Renter":2,"Renter":1,"Unknown":0}).reindex(hh)
        out["kids_code"] = dm.kid_category_desc.map({"None/Unknown":0,"1":1,"2":2,"3+":3}).reindex(hh)
        out["has_demo"] = pd.Series(dm.index.isin(hh).astype(float), index=dm.index).reindex(hh).fillna(0)
    out["snapshot_day"] = float(day)
    wk = (day + 8) // 7
    out["week_of_year"] = float(wk)
    out["week_sin"] = np.sin(2*np.pi*wk/52); out["week_cos"] = np.cos(2*np.pi*wk/52)
    return out

t0=time.time()
f = agent_api.build_features(feats)
print("built", f.shape, f"({time.time()-t0:.0f}s)")
f = f.reset_index()
agent_api.save_table(f, "e002_features")
feats = [c for c in f.columns if c not in ("household_key","snapshot_day")]
f[feats] = f[feats].astype(float)

t = agent_api.train_targets()
d = f.merge(t, on=["household_key","snapshot_day"], how="inner")
print("train rows", len(d))

params = dict(n_estimators=1200, learning_rate=0.03, max_depth=6, min_child_weight=5,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
              objective="reg:squarederror", tree_method="hist", n_jobs=4)
# internal sanity check: last train snapshot as holdout
ho = d[d.snapshot_day==431]; tr = d[d.snapshot_day<431]
m0 = xgb.XGBRegressor(**params).fit(tr[feats], tr.future_spend_4w)
print("holdout 431 MAE", np.abs(m0.predict(ho[feats])-ho.future_spend_4w).mean().round(3))

model = xgb.XGBRegressor(**params).fit(d[feats], d.future_spend_4w)
val = f[f.snapshot_day>=459].copy()
val["prediction"] = model.predict(val[feats])
print(val.prediction.describe().round(2))
p = val[["household_key","snapshot_day","prediction"]]
agent_api.save_table(p, "e002_preds")
print("saved")
