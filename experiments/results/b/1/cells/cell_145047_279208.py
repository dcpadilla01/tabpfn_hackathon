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