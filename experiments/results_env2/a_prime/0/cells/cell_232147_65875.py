import agent_api, pandas as pd, numpy as np

v = agent_api.snapshot(459)
tx = v.table("transactions")
prods = v.table("products")[["product_id","department"]]
first_day = tx.groupby("household_key").day.min()
tt = agent_api.train_targets()
E = agent_api.load_saved("e012_outcome2.parquet")

rows = []
for sd in agent_api.snapshot_days()["train"]:
    hh = first_day[first_day <= sd-84].index
    t = tx[tx.day <= sd]
    bk = t.groupby(["household_key","basket_id"]).agg(spend=("sales_value","sum"), nlines=("sales_value","size"), day=("day","first"), tt=("trans_time","first")).reset_index()
    w = lambda d0: (t.day > sd-d0)
    agg = {}
    for d0,nm in [(7,"7d"),(14,"14d")]:
        g = t[w(d0)].groupby("household_key").agg(spend=("sales_value","sum"), nb=("basket_id","nunique"))
        agg["spend_"+nm] = g.spend; agg["nbask_"+nm] = g.nb
    st28 = t[w(28)].groupby("household_key").store_id.nunique(); agg["st_n_28"] = st28
    st84 = t[w(84)].groupby("household_key").store_id.nunique(); agg["st_n_84"] = st84
    sp84 = t[w(84)].groupby(["household_key","store_id"]).sales_value.sum()
    tot84 = sp84.groupby(level=0).sum()
    mainstore = sp84.reset_index().sort_values(["household_key","sales_value"]).groupby("household_key").store_id.last()
    sp28st = t[w(28)].groupby(["household_key","store_id"]).sales_value.sum().reset_index()
    mm = sp28st.merge(mainstore.rename("main_st"), on="household_key")
    agg["st_main_spend_28"] = mm[mm.store_id==mm.main_st].groupby("household_key").sales_value.sum().reindex(hh)
    pr28 = t[w(28)].groupby("household_key").product_id.nunique(); agg["pr_n_28"] = pr28
    pr84 = t[w(84)].groupby("household_key").product_id.nunique(); agg["pr_n_84"] = pr84
    rec_pairs = t[w(84)][["household_key","product_id"]].drop_duplicates()
    pri_pairs = t[(t.day>sd-168)&(t.day<=sd-84)][["household_key","product_id"]].drop_duplicates()
    rep = rec_pairs.merge(pri_pairs, on=["household_key","product_id"], how="inner")
    agg["pr_repeat_n_84"] = rep.groupby("household_key").size()
    r84 = t[w(84)].merge(rep.assign(repv=1), on=["household_key","product_id"], how="left")
    r84["repv"] = (r84.repv==1).astype(float)*r84.sales_value
    agg["pr_repeat_spend_share_84"] = r84.groupby("household_key").repv.sum()/tot84
    pri_all = t[t.day<=sd-28][["household_key","product_id"]].drop_duplicates()
    r28 = t[w(28)].merge(pri_all.assign(k=1), on=["household_key","product_id"], how="left")
    r28["newv"] = np.where(r28["k"]!=1, r28.sales_value, 0.0)
    tot28 = t[w(28)].groupby("household_key").sales_value.sum()
    agg["pr_new_spend_share_28"] = r28.groupby("household_key").newv.sum()/tot28
    spp = t[w(84)].groupby(["household_key","product_id"]).sales_value.sum()
    agg["pr_top1_share_84"] = spp.groupby(level=0).max()/tot84
    td = t[w(84)].merge(prods, on="product_id", how="left")
    ds = td.groupby(["household_key","department"])["sales_value"].sum().reset_index()
    ds["tot"] = ds.groupby("household_key")["sales_value"].transform("sum")
    ds["p"] = ds["sales_value"]/ds["tot"]
    ds["ent"] = -ds["p"]*np.log(ds["p"].clip(lower=1e-9))
    agg["dept_entropy_84"] = ds.groupby("household_key")["ent"].sum()
    b84 = bk[bk.day > sd-84]; b28 = bk[bk.day > sd-28]
    agg["bk_mean_84"] = b84.groupby("household_key").spend.mean()
    agg["bk_std_84"] = b84.groupby("household_key").spend.std()
    agg["bk_lines_mean_28"] = b28.groupby("household_key").nlines.mean()
    agg["bk_units"] = t[w(28)].groupby("household_key").quantity.sum()/b28.groupby("household_key").size()
    agg["wknd_share_84"] = b84.assign(wk=(b84.day%7>=5).astype(float)).groupby("household_key").wk.mean()
    agg["tt_std_84"] = b84.groupby("household_key").tt.std()
    row = pd.DataFrame(agg); row["snapshot_day"] = sd
    rows.append(row.reset_index().rename(columns={"index":"household_key"}))
F = pd.concat(rows)
dupcols = [c for c in E.columns if c in F.columns and c not in ("household_key","snapshot_day")]
M = F.merge(E.drop(columns=dupcols), on=["household_key","snapshot_day"])
M = M.merge(tt, on=["household_key","snapshot_day"])
print("merged:", M.shape)

ctrl = ["te2_mean","spend_4w","te2_cv","nprod_112","spend_12w","nbask_12w","days_since_last","gap_mean_112","te2_med","exp_spend_4w"]
R = M[ctrl].rank()
A = np.c_[np.ones(len(R)), (R-R.mean()).fillna(0).values]
def pcorr(x, y, A):
    xr = pd.Series(x).rank(); yr = pd.Series(y).rank()
    xr = xr.fillna(xr.mean()).values; yr = yr.fillna(yr.mean()).values
    bx, *_ = np.linalg.lstsq(A, xr, rcond=None)
    by, *_ = np.linalg.lstsq(A, yr, rcond=None)
    return np.corrcoef(xr-A@bx, yr-A@by)[0,1]
cands = [c for c in F.columns if c not in ("household_key","snapshot_day")]
out = []
for c in cands:
    out.append((c, pcorr(M[c], M.future_spend_4w, A), M[c].notna().mean()))
out.sort(key=lambda r: -abs(r[1]))
print(f"{'feature':26s} {'pcorr':>7s} {'cover':>6s}")
for r in out: print(f"{r[0]:26s} {r[1]:7.3f} {r[2]:6.2f}")
