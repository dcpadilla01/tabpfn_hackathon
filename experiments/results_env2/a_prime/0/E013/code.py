import agent_api, pandas as pd, numpy as np
print("snapshot_days:", agent_api.snapshot_days())
v = agent_api.snapshot(95)
print("view day/week:", v.day, v.week)
hh = v.households
print("households type:", type(hh), "len:", len(hh))
try:
    print("sample:", list(hh)[:5])
except Exception as e:
    print("iter err:", e)
tx = v.table("transactions")
print("tx shape:", tx.shape, "day max:", tx.day.max())
print(tx.head(3))
tt = agent_api.train_targets()
print("targets:", tt.shape)
print(tt.head(3))
d = v.table("demographics")
print("demo shape:", d.shape)
b = agent_api.load_saved("e012_outcome2.parquet")
print("e012 shape:", b.shape)
cols = list(b.columns)
print("n cols:", len(cols))
print("first 35:", cols[:35])
print("te2/outcome cols:", [c for c in cols if 'te2' in c or 'o2' in c or 'outcome' in c])


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def probe(view, sd):
    hh = view.households
    print("day, week:", view.day, view.week, "| hh type:", type(hh), "len:", len(hh))
    print("sample hh:", list(hh)[:5])
    tx = view.table("transactions")
    print("tx shape:", tx.shape, "day range:", tx.day.min(), tx.day.max())
    print("tx cols:", list(tx.columns))
    tg = view.table("campaign_targets")
    print("campaign_targets shape:", tg.shape, "cols:", list(tg.columns))
    print("descriptions:", tg.description.value_counts().to_dict() if len(tg) else {})
    cp = view.table("coupon_redemptions")
    print("redemptions shape:", cp.shape)
    dm = view.table("display_mailer")
    print("dm shape:", dm.shape, "week max:", dm.week_no.max())
    return pd.DataFrame(index=hh)

out = agent_api.build_features(probe)
print("out shape:", out.shape)
print("snapshot days:", sorted(out.snapshot_day.unique()))


# ---- cell ----
import agent_api, pandas as pd, numpy as np
b = agent_api.load_saved("e012_outcome2.parquet")
print("shape:", b.shape)
cols = list(b.columns)
# group by prefix
import collections
groups = collections.OrderedDict()
for c in cols:
    p = c.split('_')[0] if '_' in c else c
    groups.setdefault(p, []).append(c)
for p, cs in groups.items():
    print(f"{p}: {len(cs)}", cs[:8], "..." if len(cs)>8 else "")


# ---- cell ----
import agent_api, pandas as pd, numpy as np, math

v = agent_api.snapshot(459)
tx = v.table("transactions")
prods = v.table("products")[["product_id","department"]]
first_day = tx.groupby("household_key").day.min()
tt = agent_api.train_targets()
e012 = agent_api.load_saved("e012_outcome2.parquet")
te2 = e012[["household_key","snapshot_day","te2_mean","spend_4w","te2_cv"]]

def sp_corr(a, b):
    a = pd.Series(a).rank(); b = pd.Series(b).rank()
    if a.std()==0 or b.std()==0: return np.nan
    return np.corrcoef(a, b)[0,1]

def partial(a, b, c):
    da = pd.Series(a).rank(); db = pd.Series(b).rank(); dc = pd.Series(c).rank()
    A = np.c_[np.ones(len(dc)), dc.fillna(dc.mean()).values]
    beta, *_ = np.linalg.lstsq(A, da.fillna(da.mean()).values, rcond=None)
    res = da - A@beta
    return np.corrcoef(res.fillna(0), db.fillna(db.mean()))[0,1]

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
    agg["st_main_share_84"] = sp84.groupby(level=0).max()/tot84
    mainstore = sp84.reset_index().sort_values(["household_key","sales_value"]).groupby("household_key").store_id.last()
    sp28st = t[w(28)].groupby(["household_key","store_id"]).sales_value.sum().reset_index()
    mm = sp28st.merge(mainstore.rename("main_st"), on="household_key")
    agg["st_main_spend_28"] = mm[mm.store_id==mm.main_st].groupby("household_key").sales_value.sum().reindex(hh)
    st_prior = t[(t.day>sd-112)&(t.day<=sd-28)].groupby("household_key").store_id.unique().rename("prior")
    st_recent = t[w(28)].groupby("household_key").store_id.unique().rename("rec")
    j = pd.concat([st_prior, st_recent], axis=1).dropna()
    agg["st_new_n_28"] = j.apply(lambda r: len(set(r["rec"])-set(r["prior"])), axis=1)
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
    b84 = bk[bk.day > sd-84]
    agg["bk_mean_84"] = b84.groupby("household_key").spend.mean()
    agg["bk_std_84"] = b84.groupby("household_key").spend.std()
    agg["bk_cv_84"] = agg["bk_std_84"]/agg["bk_mean_84"]
    b28 = bk[bk.day > sd-28]
    agg["bk_lines_mean_28"] = b28.groupby("household_key").nlines.mean()
    agg["bk_units"] = t[w(28)].groupby("household_key").quantity.sum()/b28.groupby("household_key").size()
    agg["wknd_share_84"] = b84.assign(wk=(b84.day%7>=5).astype(float)).groupby("household_key").wk.mean()
    agg["tt_std_84"] = b84.groupby("household_key").tt.std()
    row = pd.DataFrame(agg)
    row["snapshot_day"] = sd
    rows.append(row.reset_index().rename(columns={"index":"household_key"}))

F = pd.concat(rows)
F = F.merge(tt, on=["household_key","snapshot_day"]).merge(te2, on=["household_key","snapshot_day"])
print("rows:", len(F), "null target:", F.future_spend_4w.isna().sum())
cands = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w","te2_mean","spend_4w","te2_cv")]
out = []
for c in cands:
    x = F[c]
    out.append((c, sp_corr(x, F.future_spend_4w), partial(x, F.future_spend_4w, F.te2_mean), x.notna().mean(), x.median()))
out.sort(key=lambda r: -abs(r[2]))
print(f"{'feature':28s} {'corr':>7s} {'part':>7s} {'cover':>6s} {'med':>9s}")
for r in out: print(f"{r[0]:28s} {r[1]:7.3f} {r[2]:7.3f} {r[3]:6.2f} {r[4]:9.2f}")


# ---- cell ----
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


# ---- cell ----
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
