e1 = load_saved("e001_recent_behavior.parquet")
print("e1 cols:", e1.columns.tolist())
print("e1 shape:", e1.shape)
print(e1.head(2).T)

tt = train_targets()
print("\ntargets shape:", tt.shape)
print(tt.future_spend_4w.describe())
m = tt.merge(e1, on=["household_key","snapshot_day"], how="left")
num = m.select_dtypes(include=[np.number]).columns
print("\ncorr with target:")
print(m[num].corr()["future_spend_4w"].sort_values())

def probe(view, s):
    try:
        t = load_saved("e001_recent_behavior.parquet")
        ok = "OK " + str(t.shape)
    except NameError:
        try:
            t = agent_api.load_saved("e001_recent_behavior.parquet")
            ok = "OK-via-agent_api " + str(t.shape)
        except Exception as ex:
            ok = f"FAIL {type(ex).__name__}: {ex}"
    except Exception as ex:
        ok = f"FAIL {type(ex).__name__}: {ex}"
    print("snapshot", s, "| load_saved inside fn:", ok, "| n_hh:", len(view.households))
    return pd.DataFrame(index=pd.Index(view.households, name="household_key"))

bf = build_features(probe)
print("build_features out:", bf.shape)


# ---- cell ----
h = history(1)
h = h.sort_values("day")
print(h[["day","basket_id","sales_value"]].head(15).to_string())
print("total spend:", h.sales_value.sum())

for s in [151, 179]:
    for lo, hi, name in [(s-27, s, "spend_28 [s-27,s]"), (s-28, s-1, "spend_28 [s-28,s-1]"),
                         (s-55, s, "spend_56 [s-55,s]"), (s-56, s-1, "spend_56 [s-56,s-1]"),
                         (s-83, s, "spend_84 [s-83,s]"), (s-84, s-1, "spend_84 [s-84,s-1]"),
                         (s-55, s-28, "spend_28_prior [s-55,s-28]"), (s-56, s-29, "spend_28_prior [s-56,s-29]"),
                         (s-167, s-84, "spend_84_prior [s-167,s-84]"), (s-168, s-85, "spend_84_prior [s-168,s-85]")]:
        v = h[(h.day >= lo) & (h.day <= hi)].sales_value.sum()
        print(f"s={s} {name}: {v:.2f}")


# ---- cell ----
e2 = load_saved("e002_full.parquet")
print("e2_full cols:", e2.columns.tolist())
print("e2 shape:", e2.shape)


# ---- cell ----
E2 = load_saved("e002_full.parquet")

def fn(view, s):
    d = E2[E2.snapshot_day == s].drop(columns=["snapshot_day"]).set_index("household_key")
    d = d.reindex(pd.Index(view.households, name="household_key"))
    return d

bf = build_features(fn)
ref = load_saved("e002_full.parquet")
chk = bf.merge(ref, on=["household_key","snapshot_day"], suffixes=("_b","_r"))
cols = [c for c in ref.columns if c not in ("household_key","snapshot_day")]
diffs = {c: float(np.nanmax(np.abs(chk[c+"_b"].astype(float) - chk[c+"_r"].astype(float)))) for c in cols if np.issubdtype(ref[c].dtype, np.number)}
print("max abs diffs vs saved e002_full:", max(diffs.values()))
print("bf shape:", bf.shape)
print("NaN counts (first 5):", bf.isna().sum().sort_values(ascending=False).head(5).to_dict())


# ---- cell ----
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


# ---- cell ----
import time
E2 = load_saved("e002_full.parquet")

def num(s):
    return pd.to_numeric(s, errors="coerce")

def fn(view, s):
    hhs = pd.Index(view.households, name="household_key")
    hs = set(hhs)
    base = E2[E2.snapshot_day == s].drop(columns=["snapshot_day"]).set_index("household_key").reindex(hhs)

    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hs)].copy()
    tx["day"] = num(tx["day"])

    out = {}
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hs)]
    if len(ct):
        out["n_campaigns"] = ct.groupby("household_key").size().reindex(hhs, fill_value=0)
        piv = ct.assign(v=1).pivot_table(index="household_key", columns="description", values="v", aggfunc="max", fill_value=0)
        piv = piv.add_prefix("tgt_").reindex(hhs, fill_value=0)
        for c in piv.columns:
            out[c] = piv[c]
    else:
        out["n_campaigns"] = pd.Series(0, index=hhs)

    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hs)].copy()
    cr["day"] = num(cr["day"])
    out["redeems_all"] = cr.groupby("household_key").size().reindex(hhs, fill_value=0)
    cr84 = cr[cr.day >= s - 83]
    out["redeems_84"] = cr84.groupby("household_key").size().reindex(hhs, fill_value=0)
    out["days_since_redeem"] = (s - cr.groupby("household_key").day.max()).reindex(hhs)
    out["n_redeem_camp"] = cr.groupby("household_key").campaign.astype(str).nunique().reindex(hhs, fill_value=0)
    rd_days = cr84[["household_key", "day"]].drop_duplicates()
    if len(rd_days):
        tx_rd = tx.merge(rd_days, on=["household_key", "day"], how="inner")
        out["redeem_spend_84"] = tx_rd.groupby("household_key").sales_value.sum().reindex(hhs, fill_value=0)
    else:
        out["redeem_spend_84"] = pd.Series(0.0, index=hhs)

    t84 = tx[(tx.day >= s - 83) & (tx.day <= s)]
    w1 = (s + 8) // 7
    w0 = (s - 83 + 8) // 7
    dm = view.table("display_mailer").copy()
    dm["week_no"] = num(dm["week_no"])
    dm["display"] = num(dm["display"])
    dm["mailer"] = dm["mailer"].astype(str)
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
        mail = mg[mg.mailer != "0"]
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
print("NaNs total:", int(bf.isna().sum().sum()))
print(bf[["n_campaigns","tgt_TypeA","tgt_TypeB","tgt_TypeC","redeems_84","disp_any","disp_frac_84","mailer_touches_84"]].describe().T[["mean","50%","max"]])
path = save_table(bf, "e003_marketing.parquet")
print("saved:", path)


# ---- cell ----
import time
E2 = load_saved("e002_full.parquet")

def num(s):
    return pd.to_numeric(s, errors="coerce")

def fn(view, s):
    hhs = pd.Index(view.households, name="household_key")
    hs = set(hhs)
    base = E2[E2.snapshot_day == s].drop(columns=["snapshot_day"]).set_index("household_key").reindex(hhs)

    tx = view.table("transactions")
    tx = tx[tx.household_key.isin(hs)].copy()
    tx["day"] = num(tx["day"])

    out = {}
    ct = view.table("campaign_targets")
    ct = ct[ct.household_key.isin(hs)]
    if len(ct):
        out["n_campaigns"] = ct.groupby("household_key").size().reindex(hhs, fill_value=0)
        piv = ct.assign(v=1).pivot_table(index="household_key", columns="description", values="v", aggfunc="max", fill_value=0)
        piv = piv.add_prefix("tgt_").reindex(hhs, fill_value=0)
        for c in piv.columns:
            out[c] = piv[c]
    else:
        out["n_campaigns"] = pd.Series(0, index=hhs)

    cr = view.table("coupon_redemptions")
    cr = cr[cr.household_key.isin(hs)].copy()
    cr["day"] = num(cr["day"])
    out["redeems_all"] = cr.groupby("household_key").size().reindex(hhs, fill_value=0)
    cr84 = cr[cr.day >= s - 83]
    out["redeems_84"] = cr84.groupby("household_key").size().reindex(hhs, fill_value=0)
    out["days_since_redeem"] = (s - cr.groupby("household_key").day.max()).reindex(hhs)
    out["n_redeem_camp"] = cr.groupby("household_key").campaign.nunique().reindex(hhs, fill_value=0)
    rd_days = cr84[["household_key", "day"]].drop_duplicates()
    if len(rd_days):
        tx_rd = tx.merge(rd_days, on=["household_key", "day"], how="inner")
        out["redeem_spend_84"] = tx_rd.groupby("household_key").sales_value.sum().reindex(hhs, fill_value=0)
    else:
        out["redeem_spend_84"] = pd.Series(0.0, index=hhs)

    t84 = tx[(tx.day >= s - 83) & (tx.day <= s)]
    w1 = (s + 8) // 7
    w0 = (s - 83 + 8) // 7
    dm = view.table("display_mailer").copy()
    dm["week_no"] = num(dm["week_no"])
    dm["display"] = num(dm["display"])
    dm["mailer"] = dm["mailer"].astype(str)
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
        mail = mg[mg.mailer != "0"]
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
print("NaNs total:", int(bf.isna().sum().sum()))
print(bf[["n_campaigns","tgt_TypeA","tgt_TypeB","tgt_TypeC","redeems_84","disp_any","disp_frac_84","mailer_touches_84"]].describe().T[["mean","50%","max"]])
path = save_table(bf, "e003_marketing.parquet")
print("saved:", path)
