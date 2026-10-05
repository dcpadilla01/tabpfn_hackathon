import numpy as np, pandas as pd

e006 = agent_api.load_saved("e006_dynamics.parquet")
print("e006 shape:", e006.shape)

def style_fn(view, snapshot_day):
    day = int(snapshot_day)
    t = view.table("transactions")
    h = view.households
    if hasattr(h, "household_key"):
        hh = pd.Index(pd.unique(np.asarray(h["household_key"])))
    else:
        hh = pd.Index(np.asarray(h))
    out = pd.DataFrame(index=hh)

    w = t[t.day > day - 84]

    # basket-level stats (84d)
    b = w.groupby(["household_key", "basket_id"]).agg(
        sp=("sales_value", "sum"),
        it=("quantity", "sum"),
        npr=("product_id", "nunique"),
        tt=("trans_time", "first"),
    ).reset_index()
    ph = b.groupby("household_key")
    out["basket_sp_mean_84"] = ph["sp"].mean()
    out["basket_sp_max_84"] = ph["sp"].max()
    out["basket_sp_std_84"] = ph["sp"].std()
    out["items_per_trip_84"] = ph["it"].mean()
    out["nprod_per_trip_84"] = ph["npr"].mean()

    b["morn"] = (b.tt < 1200).astype(float)
    b["eve"] = (b.tt >= 1700).astype(float)
    me = b.groupby("household_key")[["morn", "eve"]].mean()
    out["morn_share_84"] = me["morn"]
    out["eve_share_84"] = me["eve"]
    out["trip_time_mean_84"] = ph["tt"].mean()

    # day-of-week spend shares (84d)
    w2 = w.assign(dow=w.day % 7)
    dsp = w2.groupby(["household_key", "dow"])["sales_value"].sum().unstack(fill_value=0.0)
    dsp = dsp.reindex(columns=range(7), fill_value=0.0)
    tot = dsp.sum(axis=1).replace(0, np.nan)
    for dnum in range(7):
        out[f"dow{dnum}_sp_share_84"] = dsp[dnum] / tot

    # store repertoire (84d)
    sst = w.groupby(["household_key", "store_id"])["sales_value"].sum()
    gsum = sst.groupby(level=0).sum()
    out["n_stores_84"] = sst.groupby(level=0).size()
    out["top_store_share_84"] = sst.groupby(level=0).max() / gsum.replace(0, np.nan)

    # breadth, unit price, deal reliance (84d)
    out["n_products_84"] = w.groupby("household_key")["product_id"].nunique()
    q = w[w.quantity > 0].groupby("household_key")[["sales_value", "quantity"]].sum()
    out["unit_price_84"] = q["sales_value"] / q["quantity"].replace(0, np.nan)
    disc = (w.retail_disc.abs() + w.coupon_disc.abs() + w.coupon_match_disc.abs())
    agg = pd.DataFrame({"sv": w.sales_value.values, "disc": disc.values},
                       index=w.household_key.values).groupby(level=0).sum()
    out["disc_ratio_84"] = agg["disc"] / (agg["sv"] + agg["disc"]).replace(0, np.nan)
    out["deal_line_share_84"] = pd.Series((disc > 0).astype(float).values,
                                          index=w.household_key.values).groupby(level=0).mean()

    # active-day share (84d)
    out["active_day_share_84"] = w.groupby("household_key")["day"].nunique() / 84.0

    # trip gap structure (365d)
    w3 = t[t.day > day - 365]
    dd = w3.groupby("household_key")["day"].apply(lambda s: np.sort(pd.unique(s)))
    def gstats(a):
        if len(a) < 2:
            return (np.nan, np.nan, np.nan)
        g = np.diff(a).astype(float)
        return (g.mean(), g.std(ddof=1) if len(g) > 1 else 0.0, g.max())
    gs = dd.apply(gstats)
    gexp = pd.DataFrame(gs.tolist(), index=gs.index,
                        columns=["gap_mean_365", "gap_std_365", "gap_max_365"])
    for c in gexp.columns:
        out[c] = gexp[c]

    # zero-spend 28d-block rate over trailing 364d (tenure-gated)
    fd = t.groupby("household_key")["day"].min().reindex(out.index)
    cnt = pd.Series(0.0, index=out.index)
    den = pd.Series(0.0, index=out.index)
    for k in range(1, 14):
        lo = day - 28 * k
        s = t[(t.day > lo) & (t.day <= lo + 28)].groupby("household_key")["sales_value"].sum().reindex(out.index)
        valid = (lo >= fd).astype(float)
        z = (s.fillna(0.0) <= 1e-9).astype(float)
        cnt = cnt + z * valid
        den = den + valid
    out["zero_block_rate_364"] = cnt / den.replace(0, np.nan)

    return out

newf = agent_api.build_features(style_fn)
feat_cols = [c for c in newf.columns if c not in ("household_key", "snapshot_day")]
print("newf shape:", newf.shape, "n_new_feats:", len(feat_cols))
print(newf[feat_cols].notna().mean().round(3).to_string())

overlap = (set(e006.columns) & set(newf.columns)) - {"household_key", "snapshot_day"}
print("overlap:", overlap)
if overlap:
    newf = newf.drop(columns=list(overlap))

m = e006.merge(newf, on=["household_key", "snapshot_day"], how="inner")
print("merged:", m.shape)
assert len(m) == len(e006) == len(newf), (len(m), len(e006), len(newf))
path = agent_api.save_table(m, "e012_style")
print("saved:", path)
