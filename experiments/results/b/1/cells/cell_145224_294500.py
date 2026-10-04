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