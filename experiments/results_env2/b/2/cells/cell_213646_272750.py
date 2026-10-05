import agent_api as api, pandas as pd, numpy as np

FRESH = {"PRODUCE","MEAT","MEAT-PCKGD","DELI","SEAFOOD","SEAFOOD-PCKGD","PASTRY","SALAD BAR","DAIRY DELI"}
TOPDEPS = ["GROCERY","DRUG GM","MEAT","PRODUCE","KIOSK-GAS","MEAT-PCKGD","DELI","PASTRY","MISC SALES TRAN","NUTRITION","SEAFOOD-PCKGD","FLORAL","COSMETICS","SALAD BAR","SEAFOOD","SPIRITS"]

def comp_features(view, snapshot_day):
    tx_all = view.table("transactions")
    fd = tx_all.groupby("household_key").day.min()
    hh = pd.Index(fd[fd <= snapshot_day - 84].index, name="household_key")
    tx = tx_all[tx_all.household_key.isin(hh)]
    prod = view.table("products")[["product_id","department","brand"]]
    tx = tx.merge(prod, on="product_id", how="left")
    tx["department"] = tx["department"].fillna("UNK")
    tx["brand"] = tx["brand"].fillna("UNK")
    out = pd.DataFrame(index=hh)
    for w in (84, 364):
        sub = tx[tx.day > snapshot_day - w]
        g = sub.groupby("household_key")
        tot = g.sales_value.sum().reindex(hh).fillna(0.0)
        dep_spend = sub.pivot_table(index="household_key", columns="department",
                                    values="sales_value", aggfunc="sum").reindex(hh).fillna(0.0)
        for d in TOPDEPS:
            if d in dep_spend.columns:
                out[f"sh_{d.lower().replace(' ','_').replace('.','')}_{w}"] = dep_spend[d] / tot.replace(0, np.nan)
        fresh = dep_spend[[c for c in dep_spend.columns if c in FRESH]].sum(axis=1)
        out[f"fresh_share_{w}"] = fresh / tot.replace(0, np.nan)
        br = sub.groupby(["household_key","brand"]).sales_value.sum().unstack().reindex(hh).fillna(0.0)
        for b in ("Private","National"):
            if b in br.columns: out[f"{b.lower()}_share_{w}"] = br[b] / tot.replace(0, np.nan)
        for c in ("coupon_disc","coupon_match_disc","retail_disc"):
            out[f"{c}_share_{w}"] = (-g[c].sum()).reindex(hh).fillna(0.0) / tot.replace(0, np.nan)
        st = sub.groupby("household_key").store_id
        out[f"n_stores_{w}"] = st.nunique().reindex(hh).fillna(0)
        stsp = sub.groupby(["household_key","store_id"]).sales_value.sum()
        out[f"top_store_share_{w}"] = stsp.groupby(level=0).max().reindex(hh).fillna(0.0) / tot.replace(0, np.nan)
        b = sub.groupby(["household_key","basket_id"]).agg(sv=("sales_value","sum"),
                                                           q=("quantity","sum"),
                                                           n=("sales_value","size"),
                                                           tt=("trans_time","first"))
        bg = b.groupby(level=0)
        out[f"lines_per_basket_{w}"] = bg.n.mean().reindex(hh)
        out[f"units_per_basket_{w}"] = bg.q.mean().reindex(hh)
        out[f"evening_basket_share_{w}"] = bg.tt.apply(lambda s: (s>=1700).mean()).reindex(hh)
        out[f"trans_time_mean_{w}"] = bg.tt.mean().reindex(hh)
        dow = sub.groupby([sub.household_key, sub.day % 7]).sales_value.sum().unstack().reindex(hh).fillna(0.0)
        out[f"top_dow_share_{w}"] = dow.max(axis=1) / tot.replace(0, np.nan)
    sub = tx[tx.day > snapshot_day - 364]
    dep_spend = sub.pivot_table(index="household_key", columns="department", values="sales_value", aggfunc="sum").reindex(hh).fillna(0.0)
    p = dep_spend.div(dep_spend.sum(axis=1).replace(0, np.nan), axis=0)
    out["dep_entropy"] = -(p * np.log(p.replace(0, np.nan))).sum(axis=1)
    out["n_depts"] = (dep_spend > 0).sum(axis=1)
    dem = view.table("demographics")
    for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]:
        out["dem_"+c] = dem.set_index("household_key")[c].reindex(hh)
    out["has_demographics"] = out["dem_classification_1"].notna().astype(int)
    ct = view.table("campaign_targets")
    if len(ct):
        ctp = ct.assign(v=1).pivot_table(index="household_key", columns="description", values="v", aggfunc="sum").reindex(hh).fillna(0)
        for c in ctp.columns: out["tgt_"+c.lower()] = ctp[c]
        out["n_campaigns_tgt"] = ct.groupby("household_key").campaign.nunique().reindex(hh).fillna(0)
    red = view.table("coupon_redemptions")
    if len(red):
        r84 = red[red.day > snapshot_day-84]; r364 = red[red.day > snapshot_day-364]
        out["redemptions_84"] = r84.groupby("household_key").size().reindex(hh).fillna(0)
        out["redemptions_364"] = r364.groupby("household_key").size().reindex(hh).fillna(0)
        out["n_red_campaigns"] = r364.groupby("household_key").campaign.nunique().reindex(hh).fillna(0)
        out["days_since_last_red"] = (snapshot_day - r364.groupby("household_key").day.max()).reindex(hh)
    return out

v = api.snapshot(459)
df = comp_features(v, 459)
print("shape:", df.shape)
print("NaN frac worst:", df.isna().mean().sort_values(ascending=False).head(6).round(3).to_dict())
print(df[["sh_grocery_364","fresh_share_364","private_share_364","retail_disc_share_364","top_store_share_364","lines_per_basket_364","top_dow_share_364","dep_entropy","n_campaigns_tgt"]].describe().round(3))
