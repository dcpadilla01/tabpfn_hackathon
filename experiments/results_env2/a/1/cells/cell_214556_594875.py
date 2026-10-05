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
