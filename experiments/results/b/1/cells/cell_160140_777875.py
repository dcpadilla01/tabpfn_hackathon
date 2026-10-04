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
