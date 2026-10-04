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