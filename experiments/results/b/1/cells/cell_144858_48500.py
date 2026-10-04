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