import pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    dm = view.table("display_mailer")
    wk = (s + 8) // 7
    hh = view.households
    tx = tx[tx.day >= s - 83][["household_key","product_id","store_id","day","week_no","sales_value"]]
    dm = dm[dm.week_no >= wk - 12][["product_id","store_id","week_no","display","mailer"]].copy()
    dm["disp"] = dm.display.astype(str).ne("0").astype(np.int8)
    dm["mail"] = dm.mailer.astype(str).ne("0").astype(np.int8)
    m = tx.merge(dm[["product_id","store_id","week_no","disp","mail"]], on=["product_id","store_id","week_no"], how="left")
    m["disp"] = m["disp"].fillna(0).astype(np.int8)
    m["mail"] = m["mail"].fillna(0).astype(np.int8)
    m["spend_d"] = m.sales_value * m["disp"]
    m["spend_m"] = m.sales_value * m["mail"]

    def agg(mm, tag):
        g = mm.groupby("household_key").agg(
            disp_lines=("disp","sum"), mail_lines=("mail","sum"),
            disp_spend=("spend_d","sum"), mail_spend=("spend_m","sum"),
            lines=("disp","size"),
            ndisp_prod=("product_id", lambda x: x[mm.loc[x.index,"disp"]==1].nunique()),
            disp_wks=("week_no", lambda x: x[mm.loc[x.index,"disp"]==1].nunique()),
        )
        g[f"disp_share_{tag}"] = g["disp_lines"] / g["lines"].replace(0, np.nan)
        g[f"disp_spend_share_{tag}"] = g["disp_spend"] / (g["disp_spend"] + (g["lines"]*0).where(False, mm.groupby("household_key").sales_value.sum()))
        return g.drop(columns=["lines"])

    g28 = agg(m[m.day >= s-27], "28")
    g84 = agg(m[m.day >= s-83], "84")
    g84 = g84[["disp_lines","disp_share_84","disp_spend_share_84","mail_lines"]]
    g84.columns = ["disp_lines_84","disp_share_84","disp_spend_share_84","mail_lines_84"]

    # recency: weeks since last exposure week (matched tx)
    last = m[m.disp==1].groupby("household_key").week_no.max()
    lastm = m[m.mail==1].groupby("household_key").week_no.max()
    rec = pd.DataFrame({"wks_since_disp": wk-last, "wks_since_mail": wk-lastm})

    # leading indicator: displayed/mailer products THIS week among products hh bought in last 84d
    cur = dm[dm.week_no == wk]
    curd = set(cur.loc[cur.disp==1,"product_id"]); curm = set(cur.loc[cur.mail==1,"product_id"])
    bought = tx.groupby("household_key").product_id.apply(lambda x: set(x.unique()))
    n_bought = bought.apply(len)
    lead = pd.DataFrame({
        "lead_disp": bought.apply(lambda st: len(st & curd)),
        "lead_mail": bought.apply(lambda st: len(st & curm)),
        "n_bought_84": n_bought,
    })
    lead["lead_disp_share"] = lead.lead_disp / lead.n_bought_84.replace(0, np.nan)
    lead["lead_mail_share"] = lead.lead_mail / lead.n_bought_84.replace(0, np.nan)

    X = g28.join(g84, how="outer").join(rec, how="outer").join(lead, how="outer")
    X = X.add_prefix("dm_")
    X = X.reindex(hh)
    return X

new = agent_api.build_features(fn)
print(new.shape)
base = agent_api.load_saved("e013_stationary.parquet")
out = base.merge(new.reset_index(), on=["household_key","snapshot_day"], how="inner")
print(out.shape, out.isna().mean().mean().round(3))
print([c for c in out.columns if c.startswith("dm_")])
path = agent_api.save_table(out, "e016_display.parquet")
print(path)
