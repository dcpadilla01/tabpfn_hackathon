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
