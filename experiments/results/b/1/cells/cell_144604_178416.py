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