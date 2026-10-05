import agent_api, pandas as pd, numpy as np

def build_batch2(view, day):
    hh = view.households
    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hh)]
    prod = view.table("products")[["product_id","commodity_desc"]]
    m = tx.merge(prod, on="product_id", how="left")
    m["commodity_desc"] = m.commodity_desc.fillna("UNK")
    out = pd.DataFrame(index=hh)

    # 1) commodity 56d shares, top 24 by spend
    top = (m.groupby("commodity_desc").sales_value.sum()
             .sort_values(ascending=False).head(24).index)
    r = m[m.day > day-56]
    tot = r.groupby("household_key").sales_value.sum().rename("t")
    cs = r.groupby(["household_key","commodity_desc"]).sales_value.sum().reset_index()
    cs = cs[cs.commodity_desc.isin(top)]
    piv = cs.pivot_table(index="household_key", columns="commodity_desc", values="sales_value", aggfunc="sum").reindex(hh).fillna(0.0)
    piv = piv.div(tot.reindex(piv.index).replace(0, np.nan), axis=0).fillna(0.0)
    for c in top:
        out["cm56_"+re.sub(r"[^A-Za-z0-9]+","_",c)[:18]] = piv[c].reindex(hh).fillna(0.0)

    # 2) 28d commodity spend levels
    r2 = m[m.day > day-28]
    piv2 = r2.groupby(["household_key","commodity_desc"]).sales_value.sum().unstack(fill_value=0.0)
    piv2 = piv2.reindex(columns=top).reindex(hh).fillna(0.0)
    for c in top:
        out["cm28_"+re.sub(r"[^A-Za-z0-9]+","_",c)[:18]] = piv2[c].values

    # 3) cadence: days since & mean interval between trips with top-8 commodities
    top8 = g8 = None
    top8 = (m.groupby("commodity_desc").sales_value.sum()
              .sort_values(ascending=False).head(8).index)
    for c in top8:
        cc = re.sub(r"[^A-Za-z0-9]+","_",c)[:14]
        sub = m[m.commodity_desc==c]
        g = sub.groupby("household_key").day
        last = g.max().reindex(hh)
        out["cd_last_"+cc] = (day-last).fillna(999.0)
        first = g.min().reindex(hh)
        n = g.count().reindex(hh).fillna(0)
        span = (day-first).clip(lower=1)
        out["cd_int_"+cc] = (span/n.replace(0,np.nan)).fillna(999.0)

    # 4) dwell: per-basket commodity diversity (28d, 84d)
    for win, tag in [(28,"28"),(84,"84")]:
        rw = m[m.day > day-win]
        div = rw.groupby(["household_key","basket_id"]).commodity_desc.nunique()
        out["bkdiv_"+tag] = div.groupby("household_key").mean().reindex(hh).fillna(0.0)

    # 5) dwell: 84d spend share from baskets containing GROCERY (staples anchor)
    r84 = m[m.day > day-84]
    gset = set(r84[r84.commodity_desc=="GROCERY"].basket_id)
    r84 = r84.assign(in_g=r84.basket_id.isin(gset))
    s_all = r84.groupby(["household_key"]).sales_value.sum()
    s_g = r84[r84.in_g].groupby(["household_key"]).sales_value.sum()
    out["dwell_groc84"] = (s_g.reindex(hh).fillna(0.0)/s_all.reindex(hh).replace(0,np.nan)).fillna(0.0)
    # and share of baskets that are grocery-anchored
    bk_all = r84.groupby("household_key").basket_id.nunique()
    bk_g = r84[r84.in_g].groupby("household_key").basket_id.nunique()
    out["bkanchor_groc84"] = (bk_g.reindex(hh).fillna(0.0)/bk_all.reindex(hh).replace(0,np.nan)).fillna(0.0)

    return out

feats = agent_api.build_features(build_batch2)
print(feats.shape)
print([c for c in feats.columns if c not in ("household_key","snapshot_day")][:10])
print(feats.isna().mean().mean())
agent_api.save_table(feats, "e016_batch2.parquet")
