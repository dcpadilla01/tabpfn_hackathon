import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
y = tt["future_spend_4w"]
print("target rows:", len(tt), "| snapshots:", sorted(tt.snapshot_day.unique()))
print(y.describe().round(2).to_dict())
print("zero share:", round((y==0).mean(),4))

names = ["e009_macro.parquet","e008_decomp2.parquet","e001_history.parquet","e011_display.parquet","e012_full.parquet"]
tabs = {}
for nm in names:
    try:
        df = A.load_saved(nm)
        tabs[nm] = df
        print("\n==", nm, df.shape)
        print(list(df.columns))
    except Exception as e:
        print(nm, "ERR", repr(e))

def corrs(df, label, k=50):
    m = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    rows = []
    for c in df.columns:
        if c in ("household_key","snapshot_day"): continue
        x = m[c]
        ok = x.notna() & np.isfinite(x) if pd.api.types.is_numeric_dtype(x) else x.notna()
        if ok.sum() < 50: 
            rows.append((c, np.nan, np.nan, ok.mean())); continue
        if pd.api.types.is_numeric_dtype(x):
            r = np.corrcoef(x[ok].astype(float), y[ok].astype(float))[0,1]
            s = pd.Series(x[ok].astype(float)).corr(pd.Series(y[ok].astype(float)), method="spearman")
        else:
            r = np.nan; s = np.nan
        rows.append((c, r, s, ok.mean()))
    res = pd.DataFrame(rows, columns=["feat","pearson","spearman","nonnull"])
    res["absP"] = res.pearson.abs()
    res = res.sort_values("absP", ascending=False)
    print(f"\n--- correlations with target: {label} (merged {m.shape}) ---")
    print(res.head(k).round(3).to_string(index=False))
    return res

c9 = corrs(tabs["e009_macro.parquet"], "E009")
c1 = corrs(tabs["e001_history.parquet"], "E001", k=15)


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e006 = A.load_saved("e006_seq_gaps.parquet")
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
print("e006 extra cols:", [c for c in e006.columns if c not in e009.columns][:30])
snap_days_all = A.snapshot_days()["train"]
print("snapdays:", A.snapshot_days())

# --- check if e006 w* are aligned 28d windows ---
t = A.snapshot(431).transactions
e6 = e006[e006.snapshot_day==431].set_index("household_key")
wcols = [c for c in e6.columns if c.startswith("w")][:3]
print("wcols:", wcols)
hh_arr = np.asarray(A.snapshot(431).households)
day = t.day.values
for hh in list(hh_arr)[:3]:
    comp = []
    for k in range(1,4):
        m = (day > 431-28*k) & (day <= 431-28*(k-1)) & (t.household_key.values==hh)
        comp.append(round(float(t[m].sales_value.sum()),2))
    if hh in e6.index:
        print(hh, "aligned w1-3:", comp, "| e006:", [round(float(e6.loc[hh,c]),2) for c in wcols])

# --- candidate feature builder ---
def make_features(view, s):
    hh = np.asarray(view.households)
    keys = pd.Index(hh)
    t = view.transactions
    day = t.day.values; sv = t.sales_value.values; hk = t.household_key.values
    out = {}
    def ssum(mask, name):
        g = t[mask].groupby("household_key", sort=False).sales_value.sum()
        out[name] = g.reindex(keys).fillna(0.0)
    m14 = day > s-14; m21 = day > s-21; m84 = day > s-84; m28 = day > s-28
    ssum(m14,"spend_14"); ssum(m21,"spend_21"); ssum(m84,"spend_84")
    for k in range(2,8):
        ssum((day > s-28*k) & (day <= s-28*(k-1)), f"a{k}")
    ssum((day > s-364) & (day <= s-308), "ly_56")
    ssum((day > s-364) & (day <= s-280), "ly_84")
    arr = np.column_stack([out[f"a{k}"].reindex(keys).values for k in range(2,8)]).astype(float)
    kk = np.arange(2,8,dtype=float)
    out["a_std6"] = pd.Series(np.nanstd(arr,axis=1), index=keys)
    out["a_zero6"] = pd.Series((arr==0).sum(1).astype(float), index=keys)
    am = arr - np.nanmean(arr,axis=1,keepdims=True); am = np.nan_to_num(am)
    out["slope6"] = pd.Series((am*(kk-kk.mean())).sum(1)/((kk-kk.mean())**2).sum(), index=keys)
    tb = t[day > s-84]
    b = tb.groupby(["household_key","basket_id"], sort=False).sales_value.sum()
    bg = b.groupby("household_key", sort=False)
    nb84 = bg.size()
    out["avg_basket_84"] = bg.mean(); out["max_basket_84"] = bg.max()
    out["std_basket_84"] = bg.std(); out["nb_84"] = nb84.astype(float)
    out["top1_share_84"] = bg.max()/bg.sum()
    q = tb.groupby("household_key", sort=False).quantity.sum()
    out["items_pb_84"] = q/nb84
    bulk = tb[tb.quantity>=2].groupby("household_key", sort=False).sales_value.sum()
    out["bulk_share_84"] = (bulk.reindex(keys).fillna(0.0)/(out["spend_84"]+1e-9)).clip(0,1)
    t28 = t[day > s-28]
    out["nprod_28"] = t28.groupby("household_key", sort=False).product_id.nunique().reindex(keys)
    out["nb_28"] = t28.groupby("household_key", sort=False).basket_id.nunique().reindex(keys)
    sb28 = t28.groupby("household_key", sort=False).sales_value.sum()
    out["avg_basket_28"] = sb28/out["nb_28"]
    st = tb.groupby(["household_key","store_id"], sort=False).sales_value.sum()
    gs = st.groupby("household_key", sort=False)
    out["n_stores_84"] = gs.size().reindex(keys).astype(float)
    out["primary_share_84"] = (gs.max()/gs.sum()).reindex(keys)
    ttv = pd.to_numeric(tb.trans_time, errors="coerce")
    out["evening_share_84"] = tb.assign(_e=(ttv>=1700).astype(float)).groupby("household_key", sort=False)._e.mean().reindex(keys)
    wd = pd.Series(day % 7, index=t.index)
    out["weekend_share_84"] = tb.assign(_w=wd.isin([5,6]).astype(float)).groupby("household_key", sort=False)._w.mean().reindex(keys)
    fd = t.groupby("household_key", sort=False).day.min()
    out["tenure"] = (s - fd).reindex(keys).astype(float)
    dsum = (tb.coupon_disc.abs()+tb.coupon_match_disc.abs()+tb.retail_disc.abs()).groupby("household_key", sort=False).sum()
    out["disc_share_84"] = (dsum/(dsum+out["spend_84"]+1e-9)).clip(0,1)
    out["r14_84"] = out["spend_14"]/(out["spend_84"]+1e-9)
    c = view.campaigns
    act = c[(c.start_day<=s)&(c.end_day>=s)]
    out["n_active"] = pd.Series(float(len(act)), index=keys)
    dsc = act.description.astype(str).str[:5] if len(act) else pd.Series(dtype=str)
    for ty in ["TypeA","TypeB","TypeC"]:
        out["act_"+ty] = pd.Series(float((dsc==ty).sum()), index=keys)
    out["recent_start_28"] = pd.Series(float(((c.start_day<=s)&(c.start_day>s-28)).sum()), index=keys)
    ct = view.campaign_targets
    ta = ct[ct.campaign.isin(act.campaign)].household_key.unique() if len(act) else []
    out["hh_targ_active"] = pd.Series(pd.Index(keys).isin(set(ta)).astype(float), index=keys)
    r = view.coupon_redemptions
    out["redemp_28"] = r[r.day>s-28].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    out["redemp_84"] = r[r.day>s-84].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    df = pd.DataFrame(out).reindex(keys)
    return df.astype(float)

df_new = A.build_features(make_features)
print("df_new:", df_new.shape)
A.save_table(df_new, "cand_new.parquet")

new_cols = [c for c in df_new.columns if c not in ("household_key","snapshot_day")]
full = e009.merge(df_new, on=["household_key","snapshot_day"], how="inner")
print("full:", full.shape)

groups = {
 "G_basket": ["avg_basket_84","max_basket_84","std_basket_84","top1_share_84","nb_84","nb_28","avg_basket_28","items_pb_84","bulk_share_84","nprod_28"],
 "G_timing": ["evening_share_84","weekend_share_84","tenure","r14_84","spend_14","spend_21"],
 "G_store": ["n_stores_84","primary_share_84"],
 "G_mkt": ["n_active","act_TypeA","act_TypeB","act_TypeC","hh_targ_active","recent_start_28","redemp_28","redemp_84"],
 "G_ly": ["ly_56","ly_84"],
 "G_win": ["a2","a3","a4","a5","a6","a7","a_std6","a_zero6","slope6"],
 "G_disc": ["disc_share_84"],
}
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
rows=[]
for c in new_cols:
    x = d[c].astype(float); ok = x.notna()
    if ok.sum()>100:
        rows.append((c, np.corrcoef(x[ok], d[ycol][ok])[0,1]))
print("\nnew feat corr with target:")
print(pd.DataFrame(rows, columns=["f","r"]).sort_values("r",key=abs,ascending=False).round(3).to_string(index=False))

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0, logt=False):
    tr = d[d.snapshot_day.isin(tr_snaps)]; va = d[d.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    ytr_f = np.log1p(ytr) if logt else ytr
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr_f)
    pred = Xva@w
    if logt: pred = np.expm1(pred)
    return np.mean(np.abs(pred-yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
print("\nlocal ridge (alpha=100) MAE | holdout431(tr<=403) / holdout403(tr<=375):")
b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"BASE E009: {b1:.3f} / {b2:.3f}")
for g,cols in groups.items():
    m1 = ridge_eval(base_cols+cols, trA, 431); m2 = ridge_eval(base_cols+cols, trB, 403)
    print(f"{g:9s} +{len(cols):2d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")
allc = base_cols + sum(groups.values(), [])
m1 = ridge_eval(allc, trA, 431); m2 = ridge_eval(allc, trB, 403)
print(f"ALL      +{len(allc)-len(base_cols)}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")


# ---- cell ----
import agent_api as A, numpy as np, pandas as pd
v = A.snapshot(431)
print(type(v.households), repr(v.households)[:200])
try:
    print(len(v.households))
except Exception as e:
    print("len err", e)
print(type(v.day), repr(v.day)[:100])
print(type(v.week), repr(v.week)[:100])


# ---- cell ----
import agent_api as A
v = A.snapshot(431)
print([a for a in dir(v) if not a.startswith("_")])
print(A.describe_tables())


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e006 = A.load_saved("e006_seq_gaps.parquet")
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
snap_days_all = A.snapshot_days()["train"]

# check if e006 w* are aligned 28d windows
t = A.snapshot(431).transactions
e6 = e006[e006.snapshot_day==431].set_index("household_key")
wcols = ["w1","w2","w3"]
day = t.day.values; hkv = t.household_key.values; sv = t.sales_value.values
for hh in list(e6.index)[:3]:
    comp = []
    for k in range(1,4):
        m = (day > 431-28*k) & (day <= 431-28*(k-1)) & (hkv==hh)
        comp.append(round(float(sv[m].sum()),2))
    print(hh, "aligned w1-3:", comp, "| e006:", [round(float(e6.loc[hh,c]),2) for c in wcols])

def hh_keys(view, s):
    fd = view.transactions.groupby("household_key", sort=False).day.min()
    return fd[fd <= s-84].index

# --- candidate feature builder ---
def make_features(view, s):
    keys = pd.Index(hh_keys(view, s))
    t = view.transactions
    day = t.day.values
    out = {}
    def ssum(mask, name):
        g = t[mask].groupby("household_key", sort=False).sales_value.sum()
        out[name] = g.reindex(keys).fillna(0.0)
    ssum(day > s-14,"spend_14"); ssum(day > s-21,"spend_21"); ssum(day > s-84,"spend_84")
    for k in range(2,8):
        ssum((day > s-28*k) & (day <= s-28*(k-1)), f"a{k}")
    ssum((day > s-364) & (day <= s-308), "ly_56")
    ssum((day > s-364) & (day <= s-280), "ly_84")
    arr = np.column_stack([out[f"a{k}"].values for k in range(2,8)]).astype(float)
    kk = np.arange(2,8,dtype=float)
    out["a_std6"] = pd.Series(np.nanstd(arr,axis=1), index=keys)
    out["a_zero6"] = pd.Series((arr==0).sum(1).astype(float), index=keys)
    am = np.nan_to_num(arr - np.nanmean(arr,axis=1,keepdims=True))
    out["slope6"] = pd.Series((am*(kk-kk.mean())).sum(1)/((kk-kk.mean())**2).sum(), index=keys)
    tb = t[day > s-84]
    b = tb.groupby(["household_key","basket_id"], sort=False).sales_value.sum()
    bg = b.groupby("household_key", sort=False)
    nb84 = bg.size()
    out["avg_basket_84"] = bg.mean(); out["max_basket_84"] = bg.max()
    out["std_basket_84"] = bg.std(); out["nb_84"] = nb84.astype(float)
    out["top1_share_84"] = bg.max()/bg.sum()
    q = tb.groupby("household_key", sort=False).quantity.sum()
    out["items_pb_84"] = q/nb84
    bulk = tb[tb.quantity>=2].groupby("household_key", sort=False).sales_value.sum()
    out["bulk_share_84"] = (bulk.reindex(keys).fillna(0.0)/(out["spend_84"]+1e-9)).clip(0,1)
    t28 = t[day > s-28]
    out["nprod_28"] = t28.groupby("household_key", sort=False).product_id.nunique().reindex(keys)
    out["nb_28"] = t28.groupby("household_key", sort=False).basket_id.nunique().reindex(keys).astype(float)
    sb28 = t28.groupby("household_key", sort=False).sales_value.sum()
    out["avg_basket_28"] = sb28/out["nb_28"]
    st = tb.groupby(["household_key","store_id"], sort=False).sales_value.sum()
    gs = st.groupby("household_key", sort=False)
    out["n_stores_84"] = gs.size().reindex(keys).astype(float)
    out["primary_share_84"] = (gs.max()/gs.sum()).reindex(keys)
    ttv = pd.to_numeric(tb.trans_time, errors="coerce")
    out["evening_share_84"] = tb.assign(_e=(ttv>=1700).astype(float)).groupby("household_key", sort=False)._e.mean().reindex(keys)
    wd = pd.Series(day % 7, index=t.index)
    out["weekend_share_84"] = tb.assign(_w=wd.isin([5,6]).astype(float)).groupby("household_key", sort=False)._w.mean().reindex(keys)
    fd = t.groupby("household_key", sort=False).day.min()
    out["tenure"] = (s - fd).reindex(keys).astype(float)
    dsum = (tb.coupon_disc.abs()+tb.coupon_match_disc.abs()+tb.retail_disc.abs()).groupby("household_key", sort=False).sum()
    out["disc_share_84"] = (dsum/(dsum+out["spend_84"]+1e-9)).clip(0,1)
    out["r14_84"] = out["spend_14"]/(out["spend_84"]+1e-9)
    c = view.campaigns
    act = c[(c.start_day<=s)&(c.end_day>=s)]
    out["n_active"] = pd.Series(float(len(act)), index=keys)
    dsc = act.description.astype(str).str[:5] if len(act) else pd.Series(dtype=str)
    for ty in ["TypeA","TypeB","TypeC"]:
        out["act_"+ty] = pd.Series(float((dsc==ty).sum()), index=keys)
    out["recent_start_28"] = pd.Series(float(((c.start_day<=s)&(c.start_day>s-28)).sum()), index=keys)
    ct = view.campaign_targets
    ta = ct[ct.campaign.isin(act.campaign)].household_key.unique() if len(act) else []
    out["hh_targ_active"] = pd.Series(pd.Index(keys).isin(set(ta)).astype(float), index=keys)
    r = view.coupon_redemptions
    out["redemp_28"] = r[r.day>s-28].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    out["redemp_84"] = r[r.day>s-84].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    return pd.DataFrame(out).reindex(keys).astype(float)

df_new = A.build_features(make_features)
print("df_new:", df_new.shape)
A.save_table(df_new, "cand_new.parquet")

new_cols = [c for c in df_new.columns if c not in ("household_key","snapshot_day")]
full = e009.merge(df_new, on=["household_key","snapshot_day"], how="inner")
print("full:", full.shape)

groups = {
 "G_basket": ["avg_basket_84","max_basket_84","std_basket_84","top1_share_84","nb_84","nb_28","avg_basket_28","items_pb_84","bulk_share_84","nprod_28"],
 "G_timing": ["evening_share_84","weekend_share_84","tenure","r14_84","spend_14","spend_21"],
 "G_store": ["n_stores_84","primary_share_84"],
 "G_mkt": ["n_active","act_TypeA","act_TypeB","act_TypeC","hh_targ_active","recent_start_28","redemp_28","redemp_84"],
 "G_ly": ["ly_56","ly_84"],
 "G_win": ["a2","a3","a4","a5","a6","a7","a_std6","a_zero6","slope6"],
 "G_disc": ["disc_share_84"],
}
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
rows=[]
for c in new_cols:
    x = d[c].astype(float); ok = x.notna()
    if ok.sum()>100:
        rows.append((c, np.corrcoef(x[ok], d[ycol][ok])[0,1]))
print("\nnew feat corr with target:")
print(pd.DataFrame(rows, columns=["f","r"]).sort_values("r",key=abs,ascending=False).round(3).to_string(index=False))

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0, logt=False):
    tr = d[d.snapshot_day.isin(tr_snaps)]; va = d[d.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    ytr_f = np.log1p(ytr) if logt else ytr
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr_f)
    pred = Xva@w
    if logt: pred = np.expm1(pred)
    return np.mean(np.abs(pred-yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
print("\nlocal ridge (alpha=100) MAE | holdout431(tr<=403) / holdout403(tr<=375):")
b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"BASE E009: {b1:.3f} / {b2:.3f}")
for g,cols in groups.items():
    m1 = ridge_eval(base_cols+cols, trA, 431); m2 = ridge_eval(base_cols+cols, trB, 403)
    print(f"{g:9s} +{len(cols):2d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")
allc = base_cols + sum(groups.values(), [])
m1 = ridge_eval(allc, trA, 431); m2 = ridge_eval(allc, trB, 403)
print(f"ALL      +{len(allc)-len(base_cols)}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")


# ---- cell ----
import agent_api as A, pandas as pd, numpy as np
def dbg(view, s):
    t = view.transactions
    print("snap", s, "cols", list(t.columns)[:15], "shape", t.shape)
    print("r cols", list(view.coupon_redemptions.columns))
    print("c cols", list(view.campaigns.columns))
    print("ct cols", list(view.campaign_targets.columns))
    fd = t.groupby("household_key", sort=False).day.min()
    print("fd ok", len(fd))
    return pd.DataFrame({"x": np.zeros(len(fd))}, index=fd.index)
out = A.build_features(dbg)
print(out.shape)


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
snap_days_all = A.snapshot_days()["train"]

def make_features(view, s):
    t = view.transactions
    fd = t.groupby("household_key", sort=False).day.min()
    keys = pd.Index(fd[fd <= s-84].index)
    day = t.day.values
    out = {}
    def ssum(mask, name):
        g = t[mask].groupby("household_key", sort=False).sales_value.sum()
        out[name] = g.reindex(keys).fillna(0.0)
    ssum(day > s-14,"spend_14"); ssum(day > s-21,"spend_21"); ssum(day > s-84,"spend_84")
    for k in range(2,8):
        ssum((day > s-28*k) & (day <= s-28*(k-1)), f"a{k}")
    ssum((day > s-364) & (day <= s-308), "ly_56")
    ssum((day > s-364) & (day <= s-280), "ly_84")
    arr = np.column_stack([out[f"a{k}"].values for k in range(2,8)]).astype(float)
    kk = np.arange(2,8,dtype=float)
    out["a_std6"] = pd.Series(np.nanstd(arr,axis=1), index=keys)
    out["a_zero6"] = pd.Series((arr==0).sum(1).astype(float), index=keys)
    am = np.nan_to_num(arr - np.nanmean(arr,axis=1,keepdims=True))
    out["slope6"] = pd.Series((am*(kk-kk.mean())).sum(1)/((kk-kk.mean())**2).sum(), index=keys)
    tb = t[day > s-84]
    b = tb.groupby(["household_key","basket_id"], sort=False).sales_value.sum()
    bg = b.groupby("household_key", sort=False)
    nb84 = bg.size()
    out["avg_basket_84"] = bg.mean(); out["max_basket_84"] = bg.max()
    out["std_basket_84"] = bg.std(); out["nb_84"] = nb84.astype(float)
    out["top1_share_84"] = bg.max()/bg.sum()
    q = tb.groupby("household_key", sort=False).quantity.sum()
    out["items_pb_84"] = q/nb84
    bulk = tb[tb.quantity>=2].groupby("household_key", sort=False).sales_value.sum()
    out["bulk_share_84"] = (bulk.reindex(keys).fillna(0.0)/(out["spend_84"]+1e-9)).clip(0,1)
    t28 = t[day > s-28]
    out["nprod_28"] = t28.groupby("household_key", sort=False).product_id.nunique().reindex(keys).astype(float)
    out["nb_28"] = t28.groupby("household_key", sort=False).basket_id.nunique().reindex(keys).astype(float)
    sb28 = t28.groupby("household_key", sort=False).sales_value.sum()
    out["avg_basket_28"] = sb28/out["nb_28"]
    st = tb.groupby(["household_key","store_id"], sort=False).sales_value.sum()
    gs = st.groupby("household_key", sort=False)
    out["n_stores_84"] = gs.size().reindex(keys).astype(float)
    out["primary_share_84"] = (gs.max()/gs.sum()).reindex(keys)
    ttv = pd.to_numeric(tb.trans_time, errors="coerce")
    out["evening_share_84"] = tb.assign(_e=(ttv>=1700).astype(float)).groupby("household_key", sort=False)._e.mean().reindex(keys)
    wd = pd.Series(day % 7, index=t.index)
    out["weekend_share_84"] = tb.assign(_w=wd.isin([5,6]).astype(float)).groupby("household_key", sort=False)._w.mean().reindex(keys)
    out["tenure"] = (s - fd).reindex(keys).astype(float)
    tb2 = tb.assign(_d=(tb.coupon_disc.abs()+tb.coupon_match_disc.abs()+tb.retail_disc.abs()))
    dsum = tb2.groupby("household_key", sort=False)._d.sum()
    out["disc_share_84"] = (dsum/(dsum+out["spend_84"]+1e-9)).clip(0,1)
    out["r14_84"] = out["spend_14"]/(out["spend_84"]+1e-9)
    c = view.campaigns
    act = c[(c.start_day<=s)&(c.end_day>=s)]
    out["n_active"] = pd.Series(float(len(act)), index=keys)
    dsc = act.description.astype(str).str[:5] if len(act) else pd.Series(dtype=str)
    for ty in ["TypeA","TypeB","TypeC"]:
        out["act_"+ty] = pd.Series(float((dsc==ty).sum()), index=keys)
    out["recent_start_28"] = pd.Series(float(((c.start_day<=s)&(c.start_day>s-28)).sum()), index=keys)
    ct = view.campaign_targets
    ta = ct[ct.campaign.isin(act.campaign)].household_key.unique() if len(act) else []
    out["hh_targ_active"] = pd.Series(pd.Index(keys).isin(set(ta)).astype(float), index=keys)
    r = view.coupon_redemptions
    out["redemp_28"] = r[r.day>s-28].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    out["redemp_84"] = r[r.day>s-84].groupby("household_key", sort=False).size().reindex(keys).fillna(0.0)
    return pd.DataFrame(out).reindex(keys).astype(float)

df_new = A.build_features(make_features)
print("df_new:", df_new.shape)
A.save_table(df_new, "cand_new.parquet")

new_cols = [c for c in df_new.columns if c not in ("household_key","snapshot_day")]
full = e009.merge(df_new, on=["household_key","snapshot_day"], how="inner")
print("full:", full.shape)

groups = {
 "G_basket": ["avg_basket_84","max_basket_84","std_basket_84","top1_share_84","nb_84","nb_28","avg_basket_28","items_pb_84","bulk_share_84","nprod_28"],
 "G_timing": ["evening_share_84","weekend_share_84","tenure","r14_84","spend_14","spend_21"],
 "G_store": ["n_stores_84","primary_share_84"],
 "G_mkt": ["n_active","act_TypeA","act_TypeB","act_TypeC","hh_targ_active","recent_start_28","redemp_28","redemp_84"],
 "G_ly": ["ly_56","ly_84"],
 "G_win": ["a_std6","a_zero6","slope6"],
 "G_disc": ["disc_share_84"],
}
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
rows=[]
for c in new_cols:
    x = d[c].astype(float); ok = x.notna()
    if ok.sum()>100:
        rows.append((c, np.corrcoef(x[ok], d[ycol][ok])[0,1]))
print("\nnew feat corr with target:")
print(pd.DataFrame(rows, columns=["f","r"]).sort_values("r",key=abs,ascending=False).round(3).to_string(index=False))

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0, logt=False):
    tr = d[d.snapshot_day.isin(tr_snaps)]; va = d[d.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    ytr_f = np.log1p(ytr) if logt else ytr
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr_f)
    pred = Xva@w
    if logt: pred = np.expm1(pred)
    return np.mean(np.abs(pred-yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
print("\nlocal ridge (alpha=100) MAE | holdout431(tr<=403) / holdout403(tr<=375):")
b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"BASE E009: {b1:.3f} / {b2:.3f}")
for g,cols in groups.items():
    m1 = ridge_eval(base_cols+cols, trA, 431); m2 = ridge_eval(base_cols+cols, trB, 403)
    print(f"{g:9s} +{len(cols):2d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")
allc = base_cols + sum(groups.values(), [])
m1 = ridge_eval(allc, trA, 431); m2 = ridge_eval(allc, trB, 403)
print(f"ALL      +{len(allc)-len(base_cols)}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]

base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
e001_cols = [c for c in e001.columns if c not in ("household_key","snapshot_day")]
cand_cols = [c for c in cand.columns if c not in ("household_key","snapshot_day")]

full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
# dedupe columns that appear twice (spend_28, nprod_28 etc.)
dupes = [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]
drop = set()
for c in dupes:
    stem = c[:-3]
    if stem in full.columns:
        a, b = full[stem].values.astype(float), full[c].values.astype(float)
        m = np.isnan(a)&np.isnan(b)
        if np.allclose(np.nan_to_num(a), np.nan_to_num(b)) or np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
        else:
            print("DIFFERS:", c)
full = full.drop(columns=list(drop))
print("full:", full.shape)

d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"

# rank features (cross-sectional within snapshot) for top features
rank_feats = {}
for c in ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]:
    if c in d.columns:
        rank_feats["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
d2 = pd.concat([d, pd.DataFrame(rank_feats)], axis=1)
rank_cols = list(rank_feats.columns)

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0):
    tr = d2[d2.snapshot_day.isin(tr_snaps)]; va = d2[d2.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]

e001_raw = [c for c in ["spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","weekly_rate_84","log_spend_all","log_spend_28"] if c in full.columns]
demo_cal = [c for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics","snapshot_day_index","week_of_year"] if c in full.columns]
best_new = ["a_std6","a_zero6","slope6","nb_84","top1_share_84","max_basket_84","ly_84","ly_56","avg_basket_84","items_pb_84","n_stores_84"]

print("\nlocal ridge MAE | holdout431 / holdout403  (BASE=E009 42 feats)")
b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"BASE          : {b1:.3f} / {b2:.3f}")
combos = {
 "+E001raw": base_cols+e001_raw,
 "+E001raw+demo/cal": base_cols+e001_raw+demo_cal,
 "+E001raw+bestnew": base_cols+e001_raw+best_new,
 "+E001raw+bestnew+demo/cal": base_cols+e001_raw+best_new+demo_cal,
 "+bestnew only": base_cols+best_new,
 "+demo/cal only": base_cols+demo_cal,
 "+ranks": base_cols+rank_cols,
 "+E001raw+ranks": base_cols+e001_raw+rank_cols,
 "+ALL": base_cols+e001_raw+best_new+demo_cal+rank_cols,
}
for name, feats in combos.items():
    m1 = ridge_eval(feats, trA, 431); m2 = ridge_eval(feats, trB, 403)
    print(f"{name:28s} n={len(feats):3d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")

# also check per-feature marginal value within BASE+E001raw via backward screen on 431
print("\nquick marginal check of best_new on top of BASE+E001raw (holdout431):")
f2 = base_cols+e001_raw
m0 = ridge_eval(f2, trA, 431)
for c in best_new:
    m = ridge_eval(f2+[c], trA, 431)
    print(f"  +{c:18s}: {m:.3f} ({m-m0:+.3f})")


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]

base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
dupes = [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]
drop = set()
for c in dupes:
    stem = c[:-3]
    if stem in full.columns:
        a = full[stem].values.astype(float); b = full[c].values.astype(float)
        if np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
        else:
            print("DIFFERS:", c)
full = full.drop(columns=list(drop))
print("full:", full.shape)

d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
rank_feats = pd.DataFrame(index=d.index)
for c in ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]:
    if c in d.columns:
        rank_feats["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
rank_cols = list(rank_feats.columns)
d2 = pd.concat([d, rank_feats], axis=1)
print("d2:", d2.shape)

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0):
    tr = d2[d2.snapshot_day.isin(tr_snaps)]; va = d2[d2.snapshot_day==val_snap]
    Xtr = tr[feats].astype(float).values.copy(); Xva = va[feats].astype(float).values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = tr[ycol].values.astype(float); yva = va[ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
e001_raw = [c for c in ["spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","weekly_rate_84","log_spend_all","log_spend_28"] if c in full.columns]
demo_cal = [c for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics","snapshot_day_index","week_of_year"] if c in full.columns]
best_new = ["a_std6","a_zero6","slope6","nb_84","top1_share_84","max_basket_84","ly_84","ly_56","avg_basket_84","items_pb_84","n_stores_84"]

b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"\nBASE E009    : {b1:.3f} / {b2:.3f}")
combos = {
 "+E001raw": base_cols+e001_raw,
 "+E001raw+demo/cal": base_cols+e001_raw+demo_cal,
 "+E001raw+bestnew": base_cols+e001_raw+best_new,
 "+E001raw+bestnew+demo": base_cols+e001_raw+best_new+demo_cal,
 "+bestnew only": base_cols+best_new,
 "+demo/cal only": base_cols+demo_cal,
 "+ranks": base_cols+rank_cols,
 "+E001raw+ranks": base_cols+e001_raw+rank_cols,
 "+ALL": base_cols+e001_raw+best_new+demo_cal+rank_cols,
}
for name, feats in combos.items():
    m1 = ridge_eval(feats, trA, 431); m2 = ridge_eval(feats, trB, 403)
    print(f"{name:28s} n={len(feats):3d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")

print("\nmarginal of best_new on BASE+E001raw (holdout431):")
f2 = base_cols+e001_raw
m0 = ridge_eval(f2, trA, 431)
for c in best_new:
    m = ridge_eval(f2+[c], trA, 431)
    print(f"  +{c:18s}: {m:.3f} ({m-m0:+.3f})")


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]

base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
dupes = [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]
drop = set()
for c in dupes:
    stem = c[:-3]
    if stem in full.columns:
        a = full[stem].values.astype(float); b = full[c].values.astype(float)
        if np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
full = full.drop(columns=list(drop))
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"

# encode categoricals/strings as numeric codes
num_cols, cat_cols = [], []
for c in full.columns:
    if c in ("household_key","snapshot_day"): continue
    if pd.api.types.is_numeric_dtype(d[c]): num_cols.append(c)
    else: cat_cols.append(c)
enc = pd.DataFrame(index=d.index)
for c in cat_cols:
    enc[c] = pd.factorize(d[c].astype(str))[0].astype(float)
enc = pd.concat([enc, d[num_cols].astype(float)], axis=1)
print("num", len(num_cols), "cat", len(cat_cols))

rank_feats = pd.DataFrame(index=d.index)
for c in ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]:
    if c in enc.columns:
        rank_feats["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
rank_cols = list(rank_feats.columns)
enc2 = pd.concat([enc, rank_feats], axis=1)

def ridge_eval(feats, tr_snaps, val_snap, alpha=100.0):
    tr = enc2.loc[enc2.index.isin(d.index[d.snapshot_day.isin(tr_snaps)])]
    va = enc2.loc[enc2.index.isin(d.index[d.snapshot_day==val_snap])]
    Xtr = tr[feats].values.copy(); Xva = va[feats].values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = d.loc[tr.index, ycol].values.astype(float); yva = d.loc[va.index, ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
e001_raw = [c for c in ["spend_7","spend_56","spend_84","spend_182","spend_365","spend_all","nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","weekly_rate_84","log_spend_all","log_spend_28"] if c in enc.columns]
demo_cal = [c for c in ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics","snapshot_day_index","week_of_year"] if c in enc.columns]
best_new = ["a_std6","a_zero6","slope6","nb_84","top1_share_84","max_basket_84","ly_84","ly_56","avg_basket_84","items_pb_84","n_stores_84"]

b1 = ridge_eval(base_cols, trA, 431); b2 = ridge_eval(base_cols, trB, 403)
print(f"\nBASE E009    : {b1:.3f} / {b2:.3f}")
combos = {
 "+E001raw": base_cols+e001_raw,
 "+E001raw+demo/cal": base_cols+e001_raw+demo_cal,
 "+E001raw+bestnew": base_cols+e001_raw+best_new,
 "+E001raw+bestnew+demo": base_cols+e001_raw+best_new+demo_cal,
 "+bestnew only": base_cols+best_new,
 "+demo/cal only": base_cols+demo_cal,
 "+ranks": base_cols+rank_cols,
 "+E001raw+ranks": base_cols+e001_raw+rank_cols,
 "+ALL": base_cols+e001_raw+best_new+demo_cal+rank_cols,
}
for name, feats in combos.items():
    m1 = ridge_eval(feats, trA, 431); m2 = ridge_eval(feats, trB, 403)
    print(f"{name:28s} n={len(feats):3d}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f})")

print("\nmarginal of best_new on BASE+E001raw (holdout431):")
f2 = base_cols+e001_raw
m0 = ridge_eval(f2, trA, 431)
for c in best_new:
    m = ridge_eval(f2+[c], trA, 431)
    print(f"  +{c:18s}: {m:.3f} ({m-m0:+.3f})")


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

tt = A.train_targets()
e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
cand = A.load_saved("cand_new.parquet")
snap_days_all = A.snapshot_days()["train"]
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]

full = e009.merge(e001, on=["household_key","snapshot_day"], how="inner", suffixes=("","_e1"))
full = full.merge(cand, on=["household_key","snapshot_day"], how="inner", suffixes=("","_c"))
drop = set()
for c in [c for c in full.columns if c.endswith("_e1") or c.endswith("_c")]:
    stem = c[:-3]
    if stem in full.columns:
        a = full[stem].values.astype(float); b = full[c].values.astype(float)
        if np.nanmax(np.abs(np.nan_to_num(a)-np.nan_to_num(b))) < 1e-6:
            drop.add(c)
full = full.drop(columns=list(drop))
d = full.merge(tt, on=["household_key","snapshot_day"])
ycol = "future_spend_4w"
num_cols, cat_cols = [], []
for c in full.columns:
    if c in ("household_key","snapshot_day"): continue
    (num_cols if pd.api.types.is_numeric_dtype(d[c]) else cat_cols).append(c)
enc = pd.DataFrame(index=d.index)
for c in cat_cols:
    enc[c] = pd.factorize(d[c].astype(str))[0].astype(float)
enc = pd.concat([enc, d[num_cols].astype(float)], axis=1)

rank_srcs = ["spend_84","spend_28","e13","e6","usual13","usual6","b75","p13","p6","hazard","dsl","rvu","spend_182","nb_84","nprod_28","gap_n21","cv13"]
rk = pd.DataFrame(index=d.index)
for c in rank_srcs:
    if c in enc.columns:
        rk["rk_"+c] = d.groupby("snapshot_day")[c].rank(pct=True)
rk_cols = list(rk.columns)
small = ["rk_spend_84","rk_spend_28","rk_e13","rk_usual13","rk_b75","rk_nb_84","rk_nprod_28"]
enc3 = pd.concat([enc, rk], axis=1)

def ridge_eval(feats, tr_snaps, val_snap, Xdf, alpha=100.0):
    tr = Xdf.loc[d.snapshot_day.isin(tr_snaps).values]
    va = Xdf.loc[(d.snapshot_day==val_snap).values]
    Xtr = tr[feats].values.copy(); Xva = va[feats].values.copy()
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr = (Xtr-mu)/sd; Xva = (Xva-mu)/sd
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xva = np.c_[np.ones(len(Xva)), Xva]
    ytr = d.loc[tr.index, ycol].values.astype(float); yva = d.loc[va.index, ycol].values.astype(float)
    A_ = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A_[0,0] -= alpha
    w = np.linalg.solve(A_, Xtr.T@ytr)
    return np.mean(np.abs(Xva@w - yva))

trA = [x for x in snap_days_all if x < 431]
trB = [x for x in snap_days_all if x < 403]
b1 = ridge_eval(base_cols, trA, 431, enc); b2 = ridge_eval(base_cols, trB, 403, enc)
print(f"BASE: {b1:.3f} / {b2:.3f} avg={(b1+b2)/2:.3f}")
for name, rc in [("+17ranks", rk_cols), ("+7ranks", small)]:
    m1 = ridge_eval(base_cols+rc, trA, 431, enc3); m2 = ridge_eval(base_cols+rc, trB, 403, enc3)
    print(f"{name}: {m1:.3f} ({m1-b1:+.3f}) / {m2:.3f} ({m2-b2:+.3f}) avg={(m1+m2)/2:.3f}")

# greedy backward pruning of base features (ranks kept fixed = 7 small ranks)
cur = list(base_cols)
curA = ridge_eval(cur+small, trA, 431, enc3); curB = ridge_eval(cur+small, trB, 403, enc3)
print(f"\nstart cur+7ranks: {curA:.3f}/{curB:.3f} avg={(curA+curB)/2:.3f}")
for it in range(5):
    scores = []
    for c in cur:
        f = [x for x in cur if x != c]
        m1 = ridge_eval(f+small, trA, 431, enc3); m2 = ridge_eval(f+small, trB, 403, enc3)
        scores.append((c, (m1+m2)/2-(curA+curB)/2, m1-curA, m2-curB))
    sdf = pd.DataFrame(scores, columns=["c","davg","d1","d2"]).sort_values("davg")
    top = sdf.iloc[0]
    if top.davg < -0.03 and top.d1 < 0.05 and top.d2 < 0.05:
        dropped = top.c
        cur = [x for x in cur if x != dropped]
        curA = ridge_eval(cur+small, trA, 431, enc3); curB = ridge_eval(cur+small, trB, 403, enc3)
        print(f"drop {dropped:12s} -> {curA:.3f}/{curB:.3f} avg={(curA+curB)/2:.3f}")
    else:
        print("stop: no more good drops"); break
print("removed:", [c for c in base_cols if c not in cur])

final_feats = cur + small
fA = ridge_eval(final_feats, trA, 431, enc3); fB = ridge_eval(final_feats, trB, 403, enc3)
print(f"\nFINAL ({len(final_feats)} feats): {fA:.3f} / {fB:.3f} avg={(fA+fB)/2:.3f}  [BASE avg {(b1+b2)/2:.3f}]")

# build & save final table: e009 base (pruned) + within-snapshot ranks from e009 itself
out = e009[["household_key","snapshot_day"]+cur].copy()
for c in small:
    src = c[3:]
    out[c] = e009.groupby("snapshot_day")[src].rank(pct=True).values
A.save_table(out, "e013_ranks.parquet")
print("saved e013_ranks.parquet", out.shape)


# ---- cell ----
import numpy as np, pandas as pd
import agent_api as A

e009 = A.load_saved("e009_macro.parquet")
e001 = A.load_saved("e001_history.parquet")
base_cols = [c for c in e009.columns if c not in ("household_key","snapshot_day")]
removed = ['ly_spend','ly_ratio','macro_rel','macro_fut_seas','usual13_mfs']
kept = [c for c in base_cols if c not in removed]

# merge e009 (kept) with the e001 columns needed as rank sources
src_cols = ["spend_84","spend_28","e13","usual13","b75","nb_84","nprod_28"]
out = e009[["household_key","snapshot_day"]+kept].merge(
    e001[["household_key","snapshot_day"]+src_cols], on=["household_key","snapshot_day"], how="inner")
print("merged:", out.shape)

# within-snapshot percentile ranks
for c in src_cols:
    out["rk_"+c] = out.groupby("snapshot_day")[c].rank(pct=True)
A.save_table(out, "e013_ranks.parquet")
print("saved:", out.shape, "| kept base:", len(kept), "| ranks:", len(src_cols), "| total feats:", out.shape[1]-2)
print(out.head(2).iloc[:, :12].to_string())
