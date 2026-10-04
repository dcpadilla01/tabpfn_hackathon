
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
print("E001 cols (%d):" % len(e1.columns), list(e1.columns))

tt = agent_api.train_targets()
print("\nTarget describe:\n", tt.future_spend_4w.describe())
print("zero frac:", round((tt.future_spend_4w==0).mean(),3))
print("\nBy snapshot day:\n", tt.groupby("snapshot_day").future_spend_4w.agg(["count","mean","median"]))

v = agent_api.snapshot(459)
tx = v.transactions
print("\ntx day range:", tx.day.min(), tx.day.max(), "week range:", tx.week_no.min(), tx.week_no.max())
wk = tx.groupby("week_no").sales_value.sum()
print("\nWeekly total spend (every 4th week):")
print(wk.iloc[::4].round(0))

print("\nCorr with future_spend_4w on train snapshots (lag364 = same 4wk window 1yr earlier):")
rows=[]
for d in [347, 375, 403, 431]:
    fut = tt[tt.snapshot_day==d].set_index("household_key").future_spend_4w
    l364 = tx[(tx.day>=d-363)&(tx.day<=d-336)].groupby("household_key").sales_value.sum()
    t28  = tx[(tx.day>=d-27)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    t84  = tx[(tx.day>=d-83)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    df = pd.DataFrame({"fut":fut,"lag364":l364,"t28":t28,"t84":t84}).fillna(0)
    est = df[df.lag364.notna()]
    rows.append((d, round(df.fut.corr(df.lag364),3), round(est.fut.corr(est.lag364),3),
                 round(df.fut.corr(df.t28),3), round(df.fut.corr(df.t84),3), len(df), est.shape[0]))
print("d, corr_all(lag364), corr_estab(lag364), corr(t28), corr(t84), n, n_estab:")
for r in rows: print(r)


# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
v = agent_api.snapshot(459)
tx = v.transactions
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

# check spend_ly28 definition vs my own lag364
chk = []
for d in [403, 431]:
    t = e1[e1.snapshot_day==d].set_index("household_key")
    l364 = tx[(tx.day>=d-363)&(tx.day<=d-336)].groupby("household_key").sales_value.sum()
    cmp = t[["spend_ly28","ly_avail"]].join(l364.rename("my364"))
    chk.append((d, round((cmp.spend_ly28-cmp.my364).abs().max(),4), round(cmp.ly_avail.mean(),3), round((cmp.my364>0).mean(),3)))
print("spend_ly28 == lag364? (d, maxdiff, ly_avail mean, frac my364>0):", chk)

def prep(df):
    X = df.drop(columns=["household_key","snapshot_day","future_spend_4w","index"], errors="ignore")
    num = X.select_dtypes(include=[np.number,bool]).astype(float)
    dmy = pd.get_dummies(X[[c for c in CATS if c in X.columns]].astype(str), dummy_na=True)
    return pd.concat([num, dmy.astype(float)], axis=1)

def ridge_eval(extra, alphas=(3,30,100), tr_max=403, te=431):
    df = e1 if extra is None else e1.merge(extra, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    X = prep(df); y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = X[tr].mean(), X[tr].std().replace(0,1)
    Xs = ((X-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]; ytr = y[tr]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@ytr)
        pred = np.c_[np.ones(te_m.sum()), Xs[te_m]]@w
        res[a]=round(np.abs(pred-y[te_m]).mean(),2)
    return res

print("E001-only ridge MAE on day 431 by alpha:", ridge_eval(None))
print("E001-only ridge MAE on day 403 (tr<=375):", ridge_eval(None, tr_max=375, te=403))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()

def feats_for_days(days):
    out=[]
    for d in days:
        v = agent_api.snapshot(d)
        tx = v.transactions
        hh = v.households
        e1d = e1[e1.snapshot_day==d].set_index("household_key")
        g = tx.groupby("household_key")
        ly56  = tx[(tx.day>=d-419)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
        ly28b = tx[(tx.day>=d-391)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
        ly28a = tx[(tx.day>=d-363)&(tx.day<=d-336)].groupby("household_key").sales_value.sum()
        wk = (d+8)//7
        f = pd.DataFrame(index=hh)
        f["ly28_pre"]  = ly28b.reindex(hh).fillna(0)
        f["ly28_post"] = ly28a.reindex(hh).fillna(0)
        f["ly_ratio_pre"]  = f.ly28_pre/(ly56.reindex(hh).fillna(0)+1)
        f["ly_ratio_post"] = f.ly28_post/(ly56.reindex(hh).fillna(0)+1)
        f["sin1"]=np.sin(2*np.pi*wk/52.18); f["cos1"]=np.cos(2*np.pi*wk/52.18)
        f["sin2"]=np.sin(4*np.pi*wk/52.18); f["cos2"]=np.cos(4*np.pi*wk/52.18)
        pw = tx.groupby("week_no").sales_value.sum()
        cur = pw.reindex(range(wk-3,wk+1)).fillna(0).mean()
        fut = pw.reindex(range(wk+1,wk+5)).fillna(0).mean()
        f["panel_uplift"] = (fut+1)/(cur+1)
        f["seasonal_pred"] = f.panel_uplift * e1d.spend_28.reindex(hh).fillna(0)
        f["tenure"] = (d-g.day.min()).reindex(hh).fillna(0)
        f["growth_28_84"] = e1d.spend_28.reindex(hh).fillna(0)/(e1d.spend_84.reindex(hh).fillna(0)+1)
        f["zero7"] = (e1d.spend_7.reindex(hh).fillna(0)==0).astype(float)
        t = tx[["household_key","day","sales_value"]].copy()
        t["w28"] = np.exp(-(d-t.day)/40.3); t["w84"] = np.exp(-(d-t.day)/121.1)
        f["dec28"] = (t.sales_value*t.w28).groupby(t.household_key).sum().reindex(hh).fillna(0)
        f["dec84"] = (t.sales_value*t.w84).groupby(t.household_key).sum().reindex(hh).fillna(0)
        f["dec_ratio"] = f.dec28/(f.dec84+1)
        t84 = tx[tx.day>=d-83]
        bb = t84.groupby("basket_id").agg(hh=("household_key","first"), val=("sales_value","sum"), n=("product_id","count"), day=("day","first"))
        f["avg_items"] = bb.groupby("hh").n.mean().reindex(hh)
        f["std_basket"] = bb.groupby("hh").val.std().reindex(hh)
        f["basket_gap_var"] = bb.groupby("hh").day.apply(lambda s: s.sort_values().diff().std()).reindex(hh)
        f["nstores"] = t84.groupby("household_key").store_id.nunique().reindex(hh)
        f["weekend_share"] = t84.assign(we=(t84.day%7>=5)).groupby("household_key").we.mean().reindex(hh)
        f["avg_price"] = (t84.sales_value/t84.quantity.replace(0,np.nan)).groupby(t84.household_key).mean().reindex(hh)
        f["household_key"]=f.index; f["snapshot_day"]=d
        out.append(f.reset_index(drop=True))
    return pd.concat(out)

days = sorted(set(e1.snapshot_day.unique()))
X = feats_for_days(days)
print("built", X.shape)

def ridge_eval(extra, alphas=(30,100), tr_max=403, te=431):
    df = e1.merge(extra, on=["household_key","snapshot_day"], how="left").merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

base = ridge_eval(pd.DataFrame({"household_key":e1.household_key,"snapshot_day":e1.snapshot_day}))
print("base:", base)
for cols in [["ly28_pre","ly28_post","ly_ratio_pre","ly_ratio_post"],
             ["sin1","cos1","sin2","cos2","panel_uplift","seasonal_pred"],
             ["tenure","growth_28_84","zero7"],
             ["dec28","dec84","dec_ratio"],
             ["avg_items","std_basket","basket_gap_var","nstores","weekend_share","avg_price"]]:
    print(cols, "->", ridge_eval(X[["household_key","snapshot_day"]+cols]))
print("ALL ->", ridge_eval(X.drop(columns=["hh"])))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()

def feats_for_days(days):
    out=[]
    for d in days:
        v = agent_api.snapshot(d)
        tx = v.transactions
        hh = v.households
        e1d = e1[e1.snapshot_day==d].set_index("household_key")
        g = tx.groupby("household_key")
        ly56  = tx[(tx.day>=d-419)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
        ly28b = tx[(tx.day>=d-391)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
        ly28a = tx[(tx.day>=d-363)&(tx.day<=d-336)].groupby("household_key").sales_value.sum()
        wk = (d+8)//7
        f = pd.DataFrame(index=hh)
        f["ly28_pre"]  = ly28b.reindex(hh).fillna(0)
        f["ly28_post"] = ly28a.reindex(hh).fillna(0)
        f["ly_ratio_pre"]  = f.ly28_pre/(ly56.reindex(hh).fillna(0)+1)
        f["ly_ratio_post"] = f.ly28_post/(ly56.reindex(hh).fillna(0)+1)
        f["sin1"]=np.sin(2*np.pi*wk/52.18); f["cos1"]=np.cos(2*np.pi*wk/52.18)
        f["sin2"]=np.sin(4*np.pi*wk/52.18); f["cos2"]=np.cos(4*np.pi*wk/52.18)
        pw = tx.groupby("week_no").sales_value.sum()
        cur = pw.reindex(range(wk-3,wk+1)).fillna(0).mean()
        fut = pw.reindex(range(wk+1,wk+5)).fillna(0).mean()
        f["panel_uplift"] = (fut+1)/(cur+1)
        f["seasonal_pred"] = f.panel_uplift * e1d.spend_28.reindex(hh).fillna(0)
        f["tenure"] = (d-g.day.min()).reindex(hh).fillna(0)
        f["growth_28_84"] = e1d.spend_28.reindex(hh).fillna(0)/(e1d.spend_84.reindex(hh).fillna(0)+1)
        f["zero7"] = (e1d.spend_7.reindex(hh).fillna(0)==0).astype(float)
        t = tx[["household_key","day","sales_value"]].copy()
        t["w28"] = np.exp(-(d-t.day)/40.3); t["w84"] = np.exp(-(d-t.day)/121.1)
        f["dec28"] = (t.sales_value*t.w28).groupby(t.household_key).sum().reindex(hh).fillna(0)
        f["dec84"] = (t.sales_value*t.w84).groupby(t.household_key).sum().reindex(hh).fillna(0)
        f["dec_ratio"] = f.dec28/(f.dec84+1)
        t84 = tx[tx.day>=d-83]
        bb = t84.groupby("basket_id").agg(hh=("household_key","first"), val=("sales_value","sum"), n=("product_id","count"), day=("day","first"))
        f["avg_items"] = bb.groupby("hh").n.mean().reindex(hh)
        f["std_basket"] = bb.groupby("hh").val.std().reindex(hh)
        f["basket_gap_var"] = bb.groupby("hh").day.apply(lambda s: s.sort_values().diff().std()).reindex(hh)
        f["nstores"] = t84.groupby("household_key").store_id.nunique().reindex(hh)
        f["weekend_share"] = t84.assign(we=(t84.day%7>=5)).groupby("household_key").we.mean().reindex(hh)
        f["avg_price"] = (t84.sales_value/t84.quantity.replace(0,np.nan)).groupby(t84.household_key).mean().reindex(hh)
        f["household_key"]=f.index; f["snapshot_day"]=d
        out.append(f.reset_index(drop=True))
    return pd.concat(out)

days = sorted(set(e1.snapshot_day.unique()) & set(tt.snapshot_day.unique()))  # train days only
X = feats_for_days(days)
print("built", X.shape)

def ridge_eval(extra, alphas=(30,100), tr_max=403, te=431):
    df = e1.merge(extra, on=["household_key","snapshot_day"], how="left").merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

base = ridge_eval(pd.DataFrame({"household_key":e1.household_key,"snapshot_day":e1.snapshot_day}))
print("base:", base)
for cols in [["ly28_pre","ly28_post","ly_ratio_pre","ly_ratio_post"],
             ["sin1","cos1","sin2","cos2","panel_uplift","seasonal_pred"],
             ["tenure","growth_28_84","zero7"],
             ["dec28","dec84","dec_ratio"],
             ["avg_items","std_basket","basket_gap_var","nstores","weekend_share","avg_price"]]:
    print(cols, "->", ridge_eval(X[["household_key","snapshot_day"]+cols]))
print("ALL ->", ridge_eval(X.drop(columns=["hh"])))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
e2 = agent_api.load_saved("e002_marketing.parquet")
e3 = agent_api.load_saved("e003_dept_mix.parquet")
e4 = agent_api.load_saved("e004_long_hist.parquet")
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431):
    df = e1
    for t in tables: df = df.merge(t, on=["household_key","snapshot_day"], how="left", suffixes=("","_dup"))
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

print("calibration (local day431 vs official):")
print("  E000:", ridge_eval([]), "(official 92.5)")
print("  E001:", ridge_eval([]), "(official 63.0)")
print("  E001+E002:", ridge_eval([e2.drop(columns=[c for c in e2.columns if c in e1.columns and c not in ['household_key','snapshot_day']])]), "(official 64.7)")
print("  E001+E004:", ridge_eval([e4.drop(columns=[c for c in e4.columns if c in e1.columns and c not in ['household_key','snapshot_day']])]), "(official 64.0)")

# new feature ideas, computed from snapshots (train days only)
days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx = v.transactions; hh = v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = pd.DataFrame(index=hh)
    # A) fitted blend will be done outside; store raw components ratios
    f["yoy"] = e1d.spend_28.reindex(hh).fillna(0)/(e1d.spend_ly28.reindex(hh).fillna(0)+5)
    # B) store wealth: store mean basket value over 84d (all households)
    t84 = tx[tx.day>=d-83]
    sb = t84.groupby("basket_id").agg(store=("store_id","first"), val=("sales_value","sum"))
    store_wealth = sb.groupby("store").val.mean()
    hh_stores = t84.groupby("household_key").store_id
    f["store_wealth_w"] = hh_stores.apply(lambda s: store_wealth.reindex(s).mean()).reindex(hh)
    top_store = hh_stores.apply(lambda s: s.value_counts().idxmax() if len(s)>0 else np.nan)
    f["store_wealth_top"] = store_wealth.reindex(top_store).values
    # C) peer-group spend: mean spend_28 by classification_1 group (from e1 at same day, past-only info)
    grp = e1d.groupby("classification_1").spend_28.mean()
    f["peer_spend_c1"] = e1d.classification_1.map(grp).reindex(hh)
    grp5 = e1d.groupby("classification_5").spend_28.mean()
    f["peer_spend_c5"] = e1d.classification_5.map(grp5).reindex(hh)
    # D) volatility: std of weekly spend over last 12 weeks
    t = tx[tx.day>=d-83].copy(); t["wk"]=t.week_no
    wsp = t.groupby(["household_key","wk"]).sales_value.sum()
    f["wk_spend_std"] = wsp.groupby("household_key").std().reindex(hh)
    f["wk_spend_cv"] = (wsp.groupby("household_key").std()/(wsp.groupby("household_key").mean()+1)).reindex(hh)
    # E) active weeks breadth
    f["active_wk_12"] = wsp.groupby("household_key").size().reindex(hh)
    # F) interactions
    f["sp28_x_size"] = e1d.spend_28.reindex(hh).fillna(0)*pd.to_numeric(e1d.classification_4.str.extract(r"(\d)")[0], errors="coerce").reindex(hh).fillna(1)
    f["sp28_x_kids"] = e1d.spend_28.reindex(hh).fillna(0)*e1d.kid_category_desc.astype(str).isin(["1","2","3+"]).astype(float).reindex(hh)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)
print("\nnew feats built", X.shape)

for cols in [["yoy"],["store_wealth_w","store_wealth_top"],["peer_spend_c1","peer_spend_c5"],
             ["wk_spend_std","wk_spend_cv","active_wk_12"],["sp28_x_size","sp28_x_kids"]]:
    print(cols, "->", ridge_eval([X[["household_key","snapshot_day"]+cols]]))
print("ALL new ->", ridge_eval([X]))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
e2 = agent_api.load_saved("e002_marketing.parquet")
e4 = agent_api.load_saved("e004_long_hist.parquet")
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431, base=None):
    df = base if base is not None else e1
    for t in tables:
        t = t.drop(columns=[c for c in t.columns if c in df.columns and c not in ["household_key","snapshot_day"]])
        df = df.merge(t, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

# E000-like base: demographics + calendar only
e000 = e1[["household_key","snapshot_day"]+CATS+["has_demographics","snapshot_day_index","week_of_year"]]
print("calibration local day431:")
print("  E000:", ridge_eval([], base=e000), "(official 92.5)")
print("  E001:", ridge_eval([]), "(official 63.0)")
print("  E001+E002:", ridge_eval([e2]), "(official 64.7)")
print("  E001+E004:", ridge_eval([e4]), "(official 64.0)")

days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx = v.transactions; hh = v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = pd.DataFrame(index=hh)
    f["yoy"] = e1d.spend_28.reindex(hh).fillna(0)/(e1d.spend_ly28.reindex(hh).fillna(0)+5)
    t84 = tx[tx.day>=d-83]
    sb = t84.groupby("basket_id").agg(store=("store_id","first"), val=("sales_value","sum"))
    store_wealth = sb.groupby("store").val.mean()
    hh_stores = t84.groupby("household_key").store_id
    f["store_wealth_w"] = hh_stores.apply(lambda s: store_wealth.reindex(s).mean()).reindex(hh)
    top_store = hh_stores.apply(lambda s: s.value_counts().idxmax() if len(s)>0 else np.nan)
    f["store_wealth_top"] = top_store.map(store_wealth).reindex(hh)
    grp = e1d.groupby("classification_1").spend_28.mean()
    f["peer_spend_c1"] = e1d.classification_1.map(grp).reindex(hh)
    grp5 = e1d.groupby("classification_5").spend_28.mean()
    f["peer_spend_c5"] = e1d.classification_5.map(grp5).reindex(hh)
    t = tx[tx.day>=d-83].copy(); t["wk"]=t.week_no
    wsp = t.groupby(["household_key","wk"]).sales_value.sum()
    f["wk_spend_std"] = wsp.groupby("household_key").std().reindex(hh)
    f["wk_spend_cv"] = (wsp.groupby("household_key").std()/(wsp.groupby("household_key").mean()+1)).reindex(hh)
    f["active_wk_12"] = wsp.groupby("household_key").size().reindex(hh)
    c4num = pd.to_numeric(e1d.classification_4.str.extract(r"(\d)")[0], errors="coerce").reindex(hh).fillna(1)
    f["sp28_x_size"] = e1d.spend_28.reindex(hh).fillna(0)*c4num
    f["sp28_x_kids"] = e1d.spend_28.reindex(hh).fillna(0)*e1d.kid_category_desc.astype(str).isin(["1","2","3+"]).astype(float).reindex(hh)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)
print("\nnew feats built", X.shape)
for cols in [["yoy"],["store_wealth_w","store_wealth_top"],["peer_spend_c1","peer_spend_c5"],
             ["wk_spend_std","wk_spend_cv","active_wk_12"],["sp28_x_size","sp28_x_kids"]]:
    print(cols, "->", ridge_eval([X[["household_key","snapshot_day"]+cols]]))
print("ALL new ->", ridge_eval([X]))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431, base=None):
    df = base if base is not None else e1
    for t in tables:
        t = t.drop(columns=[c for c in t.columns if c in df.columns and c not in ["household_key","snapshot_day"]])
        df = df.merge(t, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

def cal_feats(tx, d, hh, e1d):
    f = pd.DataFrame(index=hh)
    sp28_now = e1d.spend_28.reindex(hh).fillna(0)
    sp84_now = e1d.spend_84.reindex(hh).fillna(0)
    # pooled pseudo-snapshots: s = d-28k, k=1..12, s>=95
    parts=[]
    for k in range(1,13):
        s = d-28*k
        if s < 95: continue
        sp = tx[(tx.day>=s-27)&(tx.day<=s)].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu}).fillna(0.0))
    pool = pd.concat(parts)
    qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,13)))
    lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
    mean_fu = pool.fu.groupby(lab).mean()
    b28 = pd.cut(sp28_now, qs, labels=False, include_lowest=True)
    f["cal_sp28"] = b28.map(mean_fu).astype(float).fillna(pool.fu.mean())
    # hinge basis of spend_28 (deterministic knots)
    for k in [10,25,50,100,200,400]:
        f[f"h_{k}"] = np.maximum(0, sp28_now-k)
    # hinge basis of spend_84
    for k in [50,150,400,800]:
        f[f"h84_{k}"] = np.maximum(0, sp84_now-k)
    return f

days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx=v.transactions; hh=v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = cal_feats(tx, d, hh, e1d)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)
print("built", X.shape)
hinge = X[["household_key","snapshot_day"]+[c for c in X.columns if c.startswith("h_") or c.startswith("h84_")]]
print("hinges only ->", ridge_eval([hinge]))
print("cal_sp28 only ->", ridge_eval([X[["household_key","snapshot_day","cal_sp28"]]]))
print("cal + hinges ->", ridge_eval([X]))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431, base=None):
    df = base if base is not None else e1
    for t in tables:
        t = t.drop(columns=[c for c in t.columns if c in df.columns and c not in ["household_key","snapshot_day"]])
        df = df.merge(t, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

def cal_feats(tx, d, hh, e1d):
    f = pd.DataFrame(index=hh)
    sp28_now = e1d.spend_28.reindex(hh).fillna(0)
    sp84_now = e1d.spend_84.reindex(hh).fillna(0)
    parts=[]
    for k in range(1,13):
        s = d-28*k
        if s < 95: continue
        sp = tx[(tx.day>=s-27)&(tx.day<=s)].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu}).fillna(0.0))
    if parts:
        pool = pd.concat(parts)
        qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,13)))
        lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
        mean_fu = pool.fu.groupby(lab).mean()
        b28 = pd.cut(sp28_now, qs, labels=False, include_lowest=True)
        f["cal_sp28"] = b28.map(mean_fu).astype(float)
    else:
        f["cal_sp28"] = np.nan
    for k in [10,25,50,100,200,400]:
        f[f"h_{k}"] = np.maximum(0, sp28_now-k)
    for k in [50,150,400,800]:
        f[f"h84_{k}"] = np.maximum(0, sp84_now-k)
    return f

days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx=v.transactions; hh=v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = cal_feats(tx, d, hh, e1d)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)
print("built", X.shape, "cal_sp28 nan:", X.cal_sp28.isna().sum())
hinge = X[["household_key","snapshot_day"]+[c for c in X.columns if c.startswith("h_") or c.startswith("h84_")]]
print("hinges only ->", ridge_eval([hinge]))
print("cal_sp28 only ->", ridge_eval([X[["household_key","snapshot_day","cal_sp28"]]]))
print("cal + hinges ->", ridge_eval([X]))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def fit_pred(tr_max=403, te=431, alpha=100, base=None):
    df = (base if base is not None else e1).merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    A = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1]); A[0,0]-=alpha
    w = np.linalg.solve(A, Xtr.T@y[tr])
    pred = np.c_[np.ones(te_m.sum()), Xs[te_m]]@w
    return pd.DataFrame({"y":y[te_m],"pred":pred})

p = fit_pred()
p["abs_err"] = (p.y-p.pred).abs()
p["bucket"] = pd.qcut(p.y, [0,.25,.5,.75,.9,.95,1], duplicates="drop")
print("MAE decomposition by true-spend bucket (day 431):")
print(p.groupby("bucket", observed=True).agg(n=("y","size"), mae=("abs_err","mean"), mean_y=("y","mean"), mean_pred=("pred","mean"), bias=("pred","mean")).round(1))
print("\ntotal MAE:", round(p.abs_err.mean(),2))
# how much of MAE comes from top 10% of PREDICTED spend?
p["pbucket"] = pd.qcut(p.pred, [0,.5,.9,.95,1], duplicates="drop")
print(p.groupby("pbucket", observed=True).agg(n=("y","size"), mae=("abs_err","mean")).round(1))
# zero-true households
z = p[p.y==0]
print("\ntrue-zero households: n=%d, MAE=%.1f, mean pred=%.1f -> share of total MAE: %.1f%%" %
      (len(z), z.abs_err.mean(), z.pred.mean(), 100*z.abs_err.sum()/p.abs_err.sum()))
# clip predictions at 0?
print("MAE with pred clipped at 0:", round((p.y-p.pred.clip(lower=0)).abs().mean(),2))
# what does optimal global scaling do?
from itertools import product
best=None
for c in [0.8,0.9,1.0,1.1]:
    m = (p.y-c*p.pred).abs().mean()
    if best is None or m<best[1]: best=(c,m)
print("best global scale on pred:", best)
# per-snapshot-day bias in train: is there a level shift?
print("\ntrain mean target by day (from earlier): means rise 130->146; check pred bias by day")
for d in [403, 431]:
    q = fit_pred(tr_max=d-28, te=d)
    print(d, "MAE:", round((q.y-q.pred).abs().mean(),2), "mean y:", round(q.y.mean(),1), "mean pred:", round(q.pred.mean(),1))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def lookup_feats(tx, d, hh, e1d):
    f = pd.DataFrame(index=hh)
    sp28 = e1d.spend_28.reindex(hh).fillna(0)
    sp84 = e1d.spend_84.reindex(hh).fillna(0)
    dsl  = e1d.days_since_last.reindex(hh).fillna(999)
    parts=[]
    for k in range(1,13):
        s = d-28*k
        if s < 95: continue
        sp = tx[(tx.day>=s-27)&(tx.day<=s)].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu}).fillna(0.0))
    pool = pd.concat(parts)
    # 1D bins on sp
    qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,11)))
    lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
    f["lk_mean"] = pd.cut(sp28, qs, labels=False, include_lowest=True).map(pool.fu.groupby(lab).mean()).astype(float)
    f["lk_median"] = pd.cut(sp28, qs, labels=False, include_lowest=True).map(pool.fu.groupby(lab).median()).astype(float)
    # 2D bins: sp deciles x recency terciles
    pool["dsl"] = 999.0  # recency at pseudo-snapshot s: days since last purchase before s
    # recompute pool with recency
    parts=[]
    for k in range(1,13):
        s = d-28*k
        if s < 95: continue
        w = tx[tx.day<=s]
        sp = w[(w.day>=s-27)].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        rl = w.groupby("household_key").day.max()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu,"dsl":s-rl}).fillna({"sp":0,"fu":0}))
    pool = pd.concat(parts)
    pool["spb"] = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
    pool["rcb"] = pd.cut(pool.dsl, [0,7,21,45,1000], labels=False)
    med2 = pool.fu.groupby([pool.spb, pool.rcb]).median()
    mean2 = pool.fu.groupby([pool.spb, pool.rcb]).mean()
    b2 = pd.DataFrame({"spb":pd.cut(sp28, qs, labels=False, include_lowest=True),
                       "rcb":pd.cut(dsl, [0,7,21,45,1000], labels=False)})
    f["lk2_median"] = [med2.get((a,b), np.nan) for a,b in zip(b2.spb, b2.rcb)]
    f["lk2_mean"]   = [mean2.get((a,b), np.nan) for a,b in zip(b2.spb, b2.rcb)]
    # global median fallback
    f = f.fillna({"lk_mean":pool.fu.mean(),"lk_median":pool.fu.median(),"lk2_median":pool.fu.median(),"lk2_mean":pool.fu.mean()})
    return f

days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx=v.transactions; hh=v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = lookup_feats(tx, d, hh, e1d)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)

# standalone MAE of lookups on day 431
m = X.merge(tt, on=["household_key","snapshot_day"])
t431 = m[m.snapshot_day==431]
for c in ["lk_mean","lk_median","lk2_mean","lk2_median"]:
    print(c, "standalone MAE day431:", round((t431[c]-t431.future_spend_4w).abs().mean(),2))

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431, base=None):
    df = base if base is not None else e1
    for t in tables:
        t = t.drop(columns=[c for c in t.columns if c in df.columns and c not in ["household_key","snapshot_day"]])
        df = df.merge(t, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

print("E001 + lk2_median ->", ridge_eval([X[["household_key","snapshot_day","lk2_median"]]]))
print("E001 + all lookups ->", ridge_eval([X]))
print("lookups only (no e1 hist) ->", ridge_eval([X], base=e1[["household_key","snapshot_day"]+CATS+["has_demographics","snapshot_day_index","week_of_year"]]))

# ---- cell ----
import pandas as pd, numpy as np

e1 = agent_api.load_saved("e001_history.parquet")
tt = agent_api.train_targets()
CATS = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc"]

def lookup_feats(tx, d, hh, e1d):
    f = pd.DataFrame(index=hh)
    sp28 = e1d.spend_28.reindex(hh).fillna(0)
    dsl  = e1d.days_since_last.reindex(hh).fillna(999)
    parts=[]
    for s in range(d-28, 27, -28):
        w = tx[tx.day<=s]
        sp = w[w.day>=s-27].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        rl = w.groupby("household_key").day.max()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu,"dsl":s-rl}).fillna({"sp":0,"fu":0}))
    pool = pd.concat(parts)
    qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,11)))
    lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
    f["lk_mean"]   = pd.cut(sp28, qs, labels=False, include_lowest=True).map(pool.fu.groupby(lab).mean()).astype(float)
    f["lk_median"] = pd.cut(sp28, qs, labels=False, include_lowest=True).map(pool.fu.groupby(lab).median()).astype(float)
    pool["spb"] = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
    pool["rcb"] = pd.cut(pool.dsl, [0,7,21,45,1000], labels=False)
    med2 = pool.fu.groupby([pool.spb, pool.rcb]).median()
    mean2 = pool.fu.groupby([pool.spb, pool.rcb]).mean()
    b2 = pd.DataFrame({"spb":pd.cut(sp28, qs, labels=False, include_lowest=True),
                       "rcb":pd.cut(dsl, [0,7,21,45,1000], labels=False)})
    f["lk2_median"] = [med2.get((a,b), np.nan) for a,b in zip(b2.spb, b2.rcb)]
    f["lk2_mean"]   = [mean2.get((a,b), np.nan) for a,b in zip(b2.spb, b2.rcb)]
    f = f.fillna({"lk_mean":pool.fu.mean(),"lk_median":pool.fu.median(),"lk2_median":pool.fu.median(),"lk2_mean":pool.fu.mean()})
    return f

days = sorted(set(tt.snapshot_day.unique()))
rows=[]
for d in days:
    v = agent_api.snapshot(d); tx=v.transactions; hh=v.households
    e1d = e1[e1.snapshot_day==d].set_index("household_key")
    f = lookup_feats(tx, d, hh, e1d)
    f["household_key"]=f.index; f["snapshot_day"]=d
    rows.append(f.reset_index(drop=True))
X = pd.concat(rows)

m = X.merge(tt, on=["household_key","snapshot_day"])
t431 = m[m.snapshot_day==431]
for c in ["lk_mean","lk_median","lk2_mean","lk2_median"]:
    print(c, "standalone MAE day431:", round((t431[c]-t431.future_spend_4w).abs().mean(),2))

def ridge_eval(tables, alphas=(30,100), tr_max=403, te=431, base=None):
    df = base if base is not None else e1
    for t in tables:
        t = t.drop(columns=[c for c in t.columns if c in df.columns and c not in ["household_key","snapshot_day"]])
        df = df.merge(t, on=["household_key","snapshot_day"], how="left")
    df = df.merge(tt, on=["household_key","snapshot_day"])
    num = df.drop(columns=["household_key","snapshot_day","future_spend_4w"]).select_dtypes(include=[np.number,bool]).astype(float)
    cats = df[CATS].astype(str)
    Xall = pd.concat([num, pd.get_dummies(cats, dummy_na=True).astype(float)], axis=1)
    y = df.future_spend_4w.values
    tr = df.snapshot_day<=tr_max; te_m = df.snapshot_day==te
    mu, sd = Xall[tr].mean(), Xall[tr].std().replace(0,1)
    Xs = ((Xall-mu)/sd).fillna(0).values
    Xtr = np.c_[np.ones(tr.sum()), Xs[tr]]
    res={}
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0]-=a
        w = np.linalg.solve(A, Xtr.T@y[tr])
        res[a]=round(np.abs(np.c_[np.ones(te_m.sum()), Xs[te_m]]@w - y[te_m]).mean(),2)
    return res

print("E001 + lk2_median ->", ridge_eval([X[["household_key","snapshot_day","lk2_median"]]]))
print("E001 + all lookups ->", ridge_eval([X]))
print("lookups only ->", ridge_eval([X], base=e1[["household_key","snapshot_day"]+CATS+["has_demographics","snapshot_day_index","week_of_year"]]))

# ---- cell ----
import pandas as pd, numpy as np

def fn(view, snapshot_day):
    d = snapshot_day
    tx = view.transactions
    hh = view.households
    f = pd.DataFrame(index=hh)

    # --- seasonal lag neighbors (LY window = [d-363, d-336] is E001's spend_ly28) ---
    ly_pre  = tx[(tx.day>=d-391)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
    ly56    = tx[(tx.day>=d-419)&(tx.day<=d-364)].groupby("household_key").sales_value.sum()
    f["ly28_pre"] = ly_pre.reindex(hh).fillna(0.0)
    f["ly_ratio_pre"] = f["ly28_pre"]/(ly56.reindex(hh).fillna(0.0)+1.0)

    # --- peer-group recent spend (demographic prior, past-only) ---
    sp28 = tx[(tx.day>=d-27)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    demo = view.demographics
    j = pd.DataFrame({"sp28": sp28}).join(demo.set_index("household_key")[["classification_1","classification_5"]])
    m1 = j.groupby("classification_1").sp28.mean(); m5 = j.groupby("classification_5").sp28.mean()
    dm = demo.set_index("household_key")
    f["peer_spend_c1"] = dm.classification_1.map(m1).reindex(hh)
    f["peer_spend_c5"] = dm.classification_5.map(m5).reindex(hh)

    # --- tenure / growth / activity flags ---
    first = tx.groupby("household_key").day.min()
    sp84 = tx[(tx.day>=d-83)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    sp7  = tx[(tx.day>=d-6)&(tx.day<=d)].groupby("household_key").sales_value.sum()
    f["tenure"] = (d - first).reindex(hh).fillna(0.0)
    f["growth_28_84"] = sp28.reindex(hh).fillna(0.0)/(sp84.reindex(hh).fillna(0.0)+1.0)
    f["zero7"] = (sp7.reindex(hh).fillna(0.0)==0).astype(float)

    # --- empirical binned-median lookup: future spend | past-28d spend, from pseudo-snapshots ---
    parts=[]
    for s in range(d-28, 27, -28):
        w = tx[tx.day<=s]
        sp = w[w.day>=s-27].groupby("household_key").sales_value.sum()
        fu = tx[(tx.day>=s+1)&(tx.day<=s+28)].groupby("household_key").sales_value.sum()
        parts.append(pd.DataFrame({"sp":sp,"fu":fu}).fillna(0.0))
    if parts:
        pool = pd.concat(parts)
        qs = np.unique(np.quantile(pool.sp, np.linspace(0,1,11)))
        lab = pd.cut(pool.sp, qs, labels=False, include_lowest=True)
        med = pool.fu.groupby(lab).median()
        f["lk_median"] = pd.cut(sp28.reindex(hh).fillna(0.0), qs, labels=False, include_lowest=True).map(med).astype(float)
        f["lk_median"] = f["lk_median"].fillna(pool.fu.median())
    else:
        f["lk_median"] = np.nan
    return f

X = agent_api.build_features(fn)
print(X.shape); print(X.head(3))
path = agent_api.save_table(X, "e005_seasonal_peer")
print(path)