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
