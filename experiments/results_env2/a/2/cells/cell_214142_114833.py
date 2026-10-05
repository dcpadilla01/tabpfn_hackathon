import numpy as np, pandas as pd, time

def make_feats(view, snapshot_day):
    tx = view.table("transactions")
    firsts = tx.groupby("household_key")["day"].min()
    hh = firsts[firsts <= snapshot_day - 84].index
    hhset = set(hh)
    tx = tx[tx.household_key.isin(hhset)]
    tx["sales_value"] = tx["sales_value"].astype(np.float64)
    out = pd.DataFrame(index=pd.Index(hh, name="household_key"))

    g = tx.groupby("household_key")
    out["spend_all"] = g["sales_value"].sum()
    out["baskets_all"] = g["basket_id"].nunique()
    out["spend_max_basket_all"] = tx.groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").max()

    for w in (14, 28, 56, 84, 112, 224):
        t = tx[tx.day > snapshot_day - w]
        gg = t.groupby("household_key")
        out[f"spend_{w}"] = gg["sales_value"].sum()
        out[f"baskets_{w}"] = gg["basket_id"].nunique()
        if w in (28, 84):
            out[f"prods_{w}"] = gg["product_id"].nunique()
            out[f"stores_{w}"] = gg["store_id"].nunique()
            out[f"qty_{w}"] = gg["quantity"].sum()
            out[f"retail_disc_{w}"] = gg["retail_disc"].sum()
            out[f"coupon_disc_{w}"] = gg["coupon_disc"].sum()
            out[f"spend_max_basket_{w}"] = t.groupby(["household_key","basket_id"])["sales_value"].sum().groupby("household_key").max()

    last = g["day"].max(); first = g["day"].min()
    out["days_since_last"] = (snapshot_day - last).clip(lower=0)
    out["days_since_first"] = (snapshot_day - first).clip(lower=0)

    t112 = tx[tx.day > snapshot_day - 112].copy()
    t112["age"] = snapshot_day - t112["day"]
    t112["dw"] = t112["sales_value"] * (0.5 ** (t112["age"] / 28.0))
    out["spend_decay_112"] = t112.groupby("household_key")["dw"].sum()

    eps = 1.0
    out["trend_28_56"] = out["spend_28"] / (out["spend_56"] + eps)
    out["trend_56_112"] = out["spend_56"] / (out["spend_112"] + eps)
    out["trend_84_224"] = out["spend_84"] / (out["spend_224"] + eps)
    out["share_recent_all"] = out["spend_28"] / (out["spend_all"] + eps)
    out["avg_basket_84"] = out["spend_84"] / (out["baskets_84"] + eps)
    out["baskets_per_day_84"] = out["baskets_84"] / 84.0
    out["active_28"] = (out["spend_28"] > 0).astype(float)

    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hhset)]
    out["n_campaigns"] = ct.groupby("household_key").size()
    pt = ct.pivot_table(index="household_key", columns="description", values="campaign", aggfunc="count")
    pt.columns = ["ct_" + str(c) for c in pt.columns]
    out = out.join(pt)

    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hhset)]
    out["redemp_all"] = cr.groupby("household_key").size()
    out["redemp_84"] = cr[cr.day > snapshot_day - 84].groupby("household_key").size()
    out["redemp_28"] = cr[cr.day > snapshot_day - 28].groupby("household_key").size()
    if len(cr):
        out["days_since_redemp"] = (snapshot_day - cr.groupby("household_key")["day"].max()).clip(lower=0)
    else:
        out["days_since_redemp"] = np.nan

    t84 = tx[tx.day > snapshot_day - 84]
    wk = (snapshot_day + 8) // 7
    dm = view.table("display_mailer")
    dm = dm[(dm.week_no > wk - 5) & (dm.week_no <= wk)]
    disp_prods = set(dm[dm.display > 0]["product_id"].unique()) if len(dm) else set()
    t28 = tx[tx.day > snapshot_day - 28]
    out["spend28_on_disp"] = t28[t28.product_id.isin(disp_prods)].groupby("household_key")["sales_value"].sum() if disp_prods else 0.0
    out["share28_on_disp"] = out["spend28_on_disp"] / (out["spend_28"] + eps)

    dem = view.table("demographics")
    if len(dem):
        dem = dem.set_index("household_key")
        for c in dem.columns:
            out["dem_" + c] = pd.Categorical(dem[c]).codes
        out["has_demo"] = out.index.isin(dem.index).astype(float)
    out["week_of_year"] = ((snapshot_day % 364) // 7)
    out = out.fillna({"n_campaigns":0,"redemp_all":0,"redemp_84":0,"redemp_28":0})
    return out

t0 = time.time()
v = agent_api.snapshot(431)
f = make_feats(v, 431)
print("time", round(time.time()-t0,1), "shape", f.shape)
print(f.isna().sum().sort_values(ascending=False).head(12))
print(f[["spend_28","spend_84","n_campaigns","spend28_on_disp","days_since_redemp"]].describe().T)
