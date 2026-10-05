import time
E2 = load_saved("e002_full.parquet")

def fn(view, s):
    hhs = pd.Index(view.households, name="household_key")
    hs = set(hhs)
    base = E2[E2.snapshot_day == s].drop(columns=["snapshot_day"]).set_index("household_key").reindex(hhs)

    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hs)]

    out = {}
    # --- campaign targeting (campaigns already started at s) ---
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hs)]
    if len(ct):
        n_camp = ct.groupby("household_key").size().reindex(hhs, fill_value=0)
        piv = ct.assign(v=1).pivot_table(index="household_key", columns="description", values="v", aggfunc="max", fill_value=0)
        piv = piv.add_prefix("tgt_").reindex(hhs, fill_value=0)
    else:
        n_camp = pd.Series(0, index=hhs)
        piv = pd.DataFrame(index=hhs)
    out["n_campaigns"] = n_camp
    for c in piv.columns:
        out[c] = piv[c]

    # --- coupon redemptions ---
    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hs)]
    out["redeems_all"] = cr.groupby("household_key").size().reindex(hhs, fill_value=0)
    cr84 = cr[cr.day >= s - 83]
    out["redeems_84"] = cr84.groupby("household_key").size().reindex(hhs, fill_value=0)
    out["days_since_redeem"] = (s - cr.groupby("household_key").day.max()).reindex(hhs)  # NaN = never
    out["n_redeem_camp"] = cr.groupby("household_key").campaign.nunique().reindex(hhs, fill_value=0)
    rd_days = cr84[["household_key", "day"]].drop_duplicates()
    if len(rd_days):
        tx_rd = tx.merge(rd_days, on=["household_key", "day"], how="inner")
        out["redeem_spend_84"] = tx_rd.groupby("household_key").sales_value.sum().reindex(hhs, fill_value=0)
    else:
        out["redeem_spend_84"] = pd.Series(0.0, index=hhs)

    # --- display / mailer exposure for products the hh bought in last 84d ---
    t84 = tx[(tx.day >= s - 83) & (tx.day <= s)]
    w1 = (s + 8) // 7
    w0 = (s - 83 + 8) // 7
    dm = view.table("display_mailer")
    dmw = dm[(dm.week_no >= w0) & (dm.week_no <= w1)]
    if len(t84) and len(dmw):
        mg = t84[["household_key", "product_id", "store_id", "week_no"]].merge(
            dmw, on=["product_id", "store_id", "week_no"], how="inner")
        disp = mg[mg.display > 0]
        out["disp_touches_84"] = disp.groupby("household_key").size().reindex(hhs, fill_value=0)
        n_prod = t84.groupby("household_key").product_id.nunique().reindex(hhs, fill_value=0)
        n_disp_prod = disp.groupby("household_key").product_id.nunique().reindex(hhs, fill_value=0)
        out["disp_frac_84"] = n_disp_prod / n_prod.replace(0, np.nan)
        out["disp_any"] = (out["disp_touches_84"] > 0).astype(int)
        mail = mg[mg.mailer.astype(str) != "0"]
        out["mailer_touches_84"] = mail.groupby("household_key").size().reindex(hhs, fill_value=0)
    else:
        for c in ["disp_touches_84", "disp_frac_84", "mailer_touches_84"]:
            out[c] = pd.Series(np.nan, index=hhs)
        out["disp_any"] = pd.Series(0, index=hhs)

    mkt = pd.DataFrame(out, index=hhs)
    return base.join(mkt, how="left")

t0 = time.time()
bf = build_features(fn)
print("build time:", round(time.time() - t0, 1), "s; shape:", bf.shape)
print("cols:", bf.columns.tolist())
print("NaNs:", bf.isna().sum().sum())
path = save_table(bf, "e003_marketing.parquet")
print("saved:", path)
