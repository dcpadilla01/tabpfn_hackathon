import agent_api, pandas as pd, numpy as np

NEWF = None  # placeholder

def fn(view, sd):
    hh = view.households
    t = view.table("transactions")
    prods = view.table("products")[["product_id","department"]]
    bk = t.groupby(["household_key","basket_id"]).agg(spend=("sales_value","sum"),
                                                      nlines=("sales_value","size"),
                                                      day=("day","first"),
                                                      tt=("trans_time","first")).reset_index()
    w = lambda d0: (t.day > sd-d0)
    agg = {}
    # fine-grained recency
    for d0, nm in [(7,"7d"),(14,"14d")]:
        g = t[w(d0)].groupby("household_key").agg(spend=("sales_value","sum"), nb=("basket_id","nunique"))
        agg["spend_"+nm] = g.spend
        agg["nbask_"+nm] = g.nb
        agg["log_spend_"+nm] = np.log1p(g.spend)
    # store choice
    st28 = t[w(28)].groupby("household_key").store_id.nunique()
    agg["st_n_28"] = st28
    sp84 = t[w(84)].groupby(["household_key","store_id"]).sales_value.sum()
    tot84 = sp84.groupby(level=0).sum()
    agg["st_main_share_84"] = (sp84.groupby(level=0).max()/tot84).replace([np.inf,-np.inf], np.nan)
    mainstore = sp84.reset_index().sort_values(["household_key","sales_value"]).groupby("household_key").store_id.last()
    sp28st = t[w(28)].groupby(["household_key","store_id"]).sales_value.sum().reset_index()
    mm = sp28st.merge(mainstore.rename("main_st"), on="household_key")
    ms28 = mm[mm.store_id==mm.main_st].groupby("household_key").sales_value.sum()
    agg["st_main_spend_28"] = ms28
    agg["log_st_main_spend_28"] = np.log1p(ms28)
    # product variety / repeat
    agg["pr_n_28"] = t[w(28)].groupby("household_key").product_id.nunique()
    rec_pairs = t[w(84)][["household_key","product_id"]].drop_duplicates()
    pri_pairs = t[(t.day>sd-168)&(t.day<=sd-84)][["household_key","product_id"]].drop_duplicates()
    rep = rec_pairs.merge(pri_pairs, on=["household_key","product_id"], how="inner")
    agg["pr_repeat_n_84"] = rep.groupby("household_key").size()
    r84 = t[w(84)].merge(rep.assign(repv=1), on=["household_key","product_id"], how="left")
    r84["repv"] = (r84.repv==1).astype(float)*r84.sales_value
    agg["pr_repeat_spend_share_84"] = (r84.groupby("household_key").repv.sum()/tot84).replace([np.inf,-np.inf], np.nan)
    pri_all = t[t.day<=sd-28][["household_key","product_id"]].drop_duplicates()
    r28 = t[w(28)].merge(pri_all.assign(k=1), on=["household_key","product_id"], how="left")
    r28["newv"] = np.where(r28["k"]!=1, r28.sales_value, 0.0)
    tot28 = t[w(28)].groupby("household_key").sales_value.sum()
    agg["pr_new_spend_share_28"] = (r28.groupby("household_key").newv.sum()/tot28).replace([np.inf,-np.inf], np.nan)
    spp = t[w(84)].groupby(["household_key","product_id"]).sales_value.sum()
    agg["pr_top1_share_84"] = (spp.groupby(level=0).max()/tot84).replace([np.inf,-np.inf], np.nan)
    # basket composition
    b84 = bk[bk.day > sd-84]; b28 = bk[bk.day > sd-28]
    agg["bk_mean_84"] = b84.groupby("household_key").spend.mean()
    agg["log_bk_mean_84"] = np.log1p(agg["bk_mean_84"])
    units28 = t[w(28)].groupby("household_key").quantity.sum()
    agg["bk_units"] = (units28/b28.groupby("household_key").size()).replace([np.inf,-np.inf], np.nan)
    agg["log_bk_units"] = np.log1p(agg["bk_units"])
    agg["wknd_share_84"] = b84.assign(wk=(b84.day%7>=5).astype(float)).groupby("household_key").wk.mean()
    # dept entropy 84d
    td = t[w(84)].merge(prods, on="product_id", how="left")
    ds = td.groupby(["household_key","department"])["sales_value"].sum().reset_index()
    ds["tot"] = ds.groupby("household_key")["sales_value"].transform("sum")
    ds["p"] = ds["sales_value"]/ds["tot"]
    ds["ent"] = -ds["p"]*np.log(ds["p"].clip(lower=1e-9))
    agg["dept_entropy_84"] = ds.groupby("household_key")["ent"].sum()
    row = pd.DataFrame(agg).reindex(hh)
    if sd in (95, 431):
        print("sd", sd, "hh", len(hh), "cols", row.shape[1], "nan%", (row.isna().mean().mean()*100).round(1))
    return row

F = agent_api.build_features(fn)
print("F shape:", F.shape)
E = agent_api.load_saved("e012_outcome2.parquet")
dup = [c for c in E.columns if c in F.columns and c not in ("household_key","snapshot_day")]
print("dup cols dropped from E012:", dup)
M = F.merge(E.drop(columns=dup), on=["household_key","snapshot_day"], how="inner")
print("M shape:", M.shape, "keys match F:", len(M)==len(F))
canon = agent_api.build_features(lambda v, s: pd.DataFrame(index=v.households))
print("canonical rows:", len(canon), "match:", set(map(tuple, canon[["household_key","snapshot_day"]].values))==set(map(tuple, M[["household_key","snapshot_day"]].values)))
print("total features:", M.shape[1]-2)
path = agent_api.save_table(M, "e013_storeprod.parquet")
print("saved:", path)
