
import pandas as pd, numpy as np

base = agent_api.load_saved("e018_merged.parquet")

def fn(view, snapshot_day):
    w = int(view.week)
    tx = view.table("transactions")
    dm = view.table("display_mailer").copy()
    dm["display"] = pd.to_numeric(dm["display"], errors="coerce").fillna(0).astype(int)
    dm["mailer"] = dm["mailer"].astype(str)
    dm["week_no"] = pd.to_numeric(dm["week_no"], errors="coerce").fillna(0).astype(int)
    tx28 = tx[tx.day >= snapshot_day - 27]

    st = tx28.groupby(["household_key", "store_id"], as_index=False).sales_value.sum()
    st.columns = ["household_key", "store_id", "s"]
    tot = st.groupby("household_key")["s"].transform("sum")
    st["sh"] = (st["s"] / tot).fillna(0.0)

    disp = dm[dm.display > 0]
    mail = dm[dm.mailer != "0"]
    disp4 = disp[disp.week_no >= w - 3]
    mail4 = mail[mail.week_no >= w - 3]

    def scnt(df, lo, hi):
        d = df[(df.week_no >= lo) & (df.week_no <= hi)]
        return d.groupby("store_id").product_id.nunique()

    for name, ser in [("c4d", scnt(disp, w-3, w)), ("c1d", scnt(disp, w, w)),
                      ("cpd", scnt(disp, w-7, w-4)), ("c4m", scnt(mail, w-3, w)),
                      ("c1m", scnt(mail, w, w)), ("cpm", scnt(mail, w-7, w-4))]:
        st[name] = st.store_id.map(ser).fillna(0.0)

    for c in ["c4d", "c4m", "c1d", "c1m", "cpd", "cpm"]:
        st["w" + c] = st.sh * st[c]
    g = st.groupby("household_key")[["wc4d", "wc4m", "wc1d", "wc1m", "wcpd", "wcpm"]].sum()
    g.columns = ["dm_disp_w4", "dm_mail_w4", "dm_disp_w1", "dm_mail_w1",
                 "dm_disp_prev4", "dm_mail_prev4"]

    idx = st.groupby("household_key")["s"].idxmax()
    main = st.loc[idx, ["household_key", "c1d", "c1m"]].set_index("household_key")
    main.columns = ["dm_disp_main_w1", "dm_mail_main_w1"]

    st["exp"] = ((st.c4d > 0) | (st.c4m > 0)).astype(float)
    g2 = st.groupby("household_key")["exp"].sum().rename("dm_stores_exp_n")

    pp = tx28.groupby(["household_key", "product_id"], as_index=False).sales_value.sum()
    pp.columns = ["household_key", "product_id", "s"]
    pd4 = disp4.groupby("product_id").store_id.nunique()
    pm4 = mail4.groupby("product_id").store_id.nunique()
    pp["dc"] = pp.product_id.map(pd4).fillna(0.0)
    pp["mc"] = pp.product_id.map(pm4).fillna(0.0)
    tp = pp.groupby("household_key")["s"].transform("sum")
    pp["dsh"] = pp.s * (pp.dc > 0) / tp
    pp["msh"] = pp.s * (pp.mc > 0) / tp
    g3 = pp.groupby("household_key").agg(dm_prod_disp_cnt=("dc", lambda s: float((s > 0).sum())),
                                         dm_prod_mail_cnt=("mc", lambda s: float((s > 0).sum())),
                                         dm_prod_disp_spend_sh=("dsh", "sum"),
                                         dm_prod_mail_spend_sh=("msh", "sum"))

    out = g.join(main).join(g2).join(g3)
    out["dm_disp_trend"] = (out.dm_disp_w4 + 1) / (out.dm_disp_prev4 + 1)
    out["dm_mail_trend"] = (out.dm_mail_w4 + 1) / (out.dm_mail_prev4 + 1)
    out["dm_disp_w1_ratio"] = (out.dm_disp_w1 + 1) / (out.dm_disp_w4 + 1)
    out["dm_mail_w1_ratio"] = (out.dm_mail_w1 + 1) / (out.dm_mail_w4 + 1)
    out["dm_any_disp"] = out.dm_disp_w4 > 0
    out["dm_any_mail"] = out.dm_mail_w4 > 0
    out = out.drop(columns=["dm_disp_prev4", "dm_mail_prev4"])
    return out.reindex(pd.Index(view.households, name="household_key")).fillna(0.0)

feats = agent_api.build_features(fn)
print("feat shape:", feats.shape)
print(feats.describe().T[["mean", "std"]].round(3))

m = base.merge(feats.drop(columns=["snapshot_day"]), on="household_key", how="left",
               suffixes=("", "_new"))
print("merged shape:", m.shape)
assert len(m) == len(base) and not m.duplicated(["household_key", "snapshot_day"]).any()
path = agent_api.save_table(m, "e020_dm.parquet")
print(path, "n_feat:", m.shape[1] - 2)
