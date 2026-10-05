import pandas as pd, numpy as np
e3 = agent_api.load_saved("e003_union.parquet")
e1 = agent_api.load_saved("e001_rfm.parquet")
print("e3", e3.shape, "e1", e1.shape)
print("E3 COLS:", sorted(map(str, e3.columns)))
t = agent_api.train_targets()
y = t["future_spend_4w"]
print("\nTARGET describe:\n", y.describe())
print("zero share %.3f median %.1f" % ((y==0).mean(), y.median()))
m = t.merge(e3, on=["household_key","snapshot_day"], how="left")
feat = [c for c in e3.columns if c not in ("household_key","snapshot_day")]
num = [c for c in feat if pd.api.types.is_numeric_dtype(m[c])]
cat = [c for c in feat if c not in num]
cor = m[num].corrwith(m["future_spend_4w"])
print("\nTOP |corr|:\n", cor.reindex(cor.abs().sort_values(ascending=False).index).head(25))
print("\nNEAR-ZERO CORR:\n", cor.reindex(cor.abs().sort_values().index).head(15))
print("\nCAT COLS:", cat)
print("\nsnapshot_days:", agent_api.snapshot_days())


# ---- cell ----
import pandas as pd, numpy as np
snap = agent_api.snapshot()  # capped at day 459
ct = snap.table("campaign_targets"); cp = snap.table("campaigns"); cr = snap.table("coupon_redemptions")
dm = snap.table("display_mailer"); tx = snap.table("transactions")
print("campaigns:\n", cp.head(15).to_string())
print("types:", cp["description"].value_counts().to_dict())
print("n campaign_targets rows", len(ct), "households", ct.household_key.nunique(), "campaigns", ct.campaign.nunique())
print("ct desc:\n", ct["description"].value_counts())
print("redemptions rows", len(cr), "households", cr.household_key.nunique(), "days", cr.day.min(), cr.day.max())
print("cr per hh describe:\n", cr.groupby("household_key").size().describe())
print("display_mailer rows", len(dm), "weeks", dm.week_no.min(), dm.week_no.max())
print("tx day range", tx.day.min(), tx.day.max(), "n hh", tx.household_key.nunique())
# overlap: how many eval households have redemptions/targets
hh_eval = set(tx.household_key.unique())
print("hh with targets: %d/%d" % (ct.household_key.nunique(), len(hh_eval)))
print("hh with redemptions: %d/%d" % (cr.household_key.nunique(), len(hh_eval)))
# how many transactions match display_mailer keys
sub = tx[tx.day >= 459-28]
w = (sub.day + 8)//7
key = sub.assign(week_no=w).merge(dm, on=["product_id","store_id","week_no"], how="left")
print("tx last28 rows", len(sub), "matched to dm:", key.display.notna().mean().round(3))


# ---- cell ----
import pandas as pd, numpy as np, time
snap = agent_api.snapshot()
dm = snap.table("display_mailer")
print("dm shape", dm.shape)
print("n products", dm.product_id.nunique(), "n stores", dm.store_id.nunique(), "n weeks", dm.week_no.nunique())
print("dup (prod,store,week):", dm.duplicated(["product_id","store_id","week_no"]).sum())
t0=time.time()
sub = dm[dm.week_no >= 50]
print("weeks>=50 rows:", len(sub), "t=%.1fs" % (time.time()-t0))
agg = sub.groupby("week_no").size()
print(agg.tail(20))
# product-week level size
pw = sub.groupby(["product_id","week_no"]).agg(nd=("store_id","nunique")).reset_index()
print("prod-week rows (weeks>=50):", len(pw))
# display/mailr non-zero share
print("display!=0 rows:", (dm.display!=0).mean().round(3), "mailer!=0:", (dm.mailer!="0").mean().round(3))
tx = snap.table("transactions")
print("tx shape", tx.shape)
t0=time.time()
w = (tx.day+8)//7
m = tx.assign(week_no=w).merge(pw, on=["product_id","week_no"], how="left")
print("merge time %.1fs, matched %.3f" % (time.time()-t0, m.nd.notna().mean()))


# ---- cell ----
import pandas as pd, numpy as np
snap = agent_api.snapshot()
dm = snap.table("display_mailer"); tx = snap.table("transactions")
print("dm dtypes:\n", dm.dtypes)
print("tx dtypes:\n", tx.dtypes)
print("display uniques sample:", dm.display.unique()[:10])
print("mailer uniques:", dm.mailer.unique())
# dup rows example
d = dm[dm.duplicated(["product_id","store_id","week_no"], keep=False)]
print("dup rows:", len(d))
print(d.head(10).to_string())
# proper match test on last 28 days
sub = tx[tx.day >= 459-28].copy()
sub["week_no"] = (sub.day+8)//7
m = sub.merge(dm, on=["product_id","store_id","week_no"], how="left")
print("last28 match rate:", m.display.notna().mean())
# ignoring store
pw = dm.groupby(["product_id","week_no"]).size().rename("n").reset_index()
m2 = sub.merge(pw, on=["product_id","week_no"], how="left")
print("last28 match ignoring store:", m2.n.notna().mean())
print("product_id dtype in tx:", sub.product_id.dtype, "in dm:", dm.product_id.dtype)


# ---- cell ----
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    ct = view.table("campaign_targets"); cp = view.table("campaigns")
    cr = view.table("coupon_redemptions"); dm = view.table("display_mailer")
    tx = view.table("transactions")
    try:
        e3 = agent_api.load_saved("e003_union.parquet")
        ok = e3.shape
    except Exception as e:
        ok = "FAIL " + type(e).__name__
    print(f"day={snapshot_day} ct={len(ct)} cp={len(cp)} cr_maxday={cr.day.max() if len(cr) else -1} "
          f"dm_maxweek={dm.week_no.max()} tx_maxday={tx.day.max()} load_saved={ok}")
    return pd.DataFrame(index=view.households[:5])

out = agent_api.build_features(fn)
print("out shape", out.shape)


# ---- cell ----
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    hh = view.households
    out = pd.DataFrame(index=hh)
    # ---------- campaigns ----------
    ct = view.table("campaign_targets"); cp = view.table("campaigns")
    if len(ct):
        ct2 = ct.drop(columns=["description"]).merge(cp, on="campaign", how="left")
        g = ct2.groupby("household_key")
        out["n_campaign_targets"] = g.size()
        out["days_since_last_tgt_start"] = snapshot_day - g["start_day"].max()
        out["tgt_active_now"] = g.apply(lambda d: bool(((d.start_day<=snapshot_day)&(d.end_day>=snapshot_day)).any()))
        out["tgt_active_future4w"] = g.apply(lambda d: bool((d.end_day>=snapshot_day+1).any()))
        for t in ["TypeA","TypeB","TypeC"]:
            out["tgt_"+t] = ct.assign(_v=ct["description"].astype(str).eq(t)).groupby("household_key")["_v"].max()
    # ---------- redemptions ----------
    cr = view.table("coupon_redemptions")
    if len(cr):
        g = cr.groupby("household_key")
        out["n_redemptions_total"] = g.size()
        out["days_since_last_redemption"] = snapshot_day - g["day"].max()
        for w in (28,56,112):
            out["n_redemptions_%dd"%w] = cr[cr.day>snapshot_day-w].groupby("household_key").size()
    # ---------- display / mailer exposure ----------
    tx = view.table("transactions")
    txr = tx[tx.day > snapshot_day-112]
    if len(txr):
        txr = txr.copy()
        txr["week_no"] = ((txr.day+8)//7).astype("int16")
        dm = view.table("display_mailer")
        dm = dm[dm.week_no >= int(txr["week_no"].min())]
        dm = dm.assign(d1=dm.display.astype(str).ne("0").values, m1=dm.mailer.astype(str).ne("0").values)
        dmk = dm.groupby(["product_id","store_id","week_no"]).agg(disp=("d1","max"), mail=("m1","max")).reset_index()
        m = txr.merge(dmk, on=["product_id","store_id","week_no"], how="left")
        m["disp"] = m["disp"].fillna(False).astype(bool); m["mail"] = m["mail"].fillna(False).astype(bool)
        for w in (28,112):
            mm = m[m.day > snapshot_day-w]
            tot = mm.groupby("household_key")["sales_value"].sum().clip(lower=1e-9)
            dv = mm.assign(_v=mm.sales_value.where(mm.disp,0.0)).groupby("household_key")["_v"].sum()
            mv = mm.assign(_v=mm.sales_value.where(mm.mail,0.0)).groupby("household_key")["_v"].sum()
            nl = mm.groupby("household_key").size()
            di = mm.assign(_v=mm.disp.astype(float)).groupby("household_key")["_v"].sum()
            bb = mm.groupby(["household_key","basket_id"])["disp"].max().groupby(level=0).mean()
            out["share_disp_%d"%w] = dv/tot
            out["share_mail_%d"%w] = mv/tot
            out["share_lines_disp_%d"%w] = di/nl
            out["share_baskets_disp_%d"%w] = bb
        cur_week = (snapshot_day+8)//7
        now_prods = set(dm.loc[(dm.week_no==cur_week) & dm.d1, "product_id"].tolist())
        mm28 = m[m.day > snapshot_day-28].copy()
        mm28["_f"] = mm28.product_id.isin(now_prods)
        tot = mm28.groupby("household_key")["sales_value"].sum().clip(lower=1e-9)
        out["share_spend_nowdisp_28"] = mm28[mm28._f].groupby("household_key")["sales_value"].sum()/tot
    return out

mk = agent_api.build_features(fn)
print("marketing table:", mk.shape)
print(mk.head(3).T)
print("\nNaN shares:\n", mk.isna().mean().round(3).to_string())
agent_api.save_table(mk.reset_index(), "e005_marketing_only.parquet")


# ---- cell ----
import pandas as pd
e3 = agent_api.load_saved("e003_union.parquet")
mk = agent_api.load_saved("e005_marketing_only.parquet")
print("e3", e3.shape, "mk", mk.shape)
m = e3.merge(mk.drop(columns=[c for c in mk.columns if c in ("snapshot_day",)] and ["snapshot_day"] if False else []), on=["household_key","snapshot_day"], how="inner")
print("merged", m.shape, "dup cols check:", len(m.columns))
agent_api.save_table(m, "e005_marketing.parquet")
