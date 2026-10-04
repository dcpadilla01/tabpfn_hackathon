import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
blocks = agent_api.load_saved("blocks_v1.parquet")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]
dem = agent_api.snapshot().demographics
d = dem.copy()
for c in ["classification_1","classification_3","classification_4","classification_5"]:
    d[c+"_num"] = d[c].astype(str).str.extract(r"(\d+)").astype(float)
d["homeowner"] = d.homeowner_desc.astype("category").cat.codes.replace(-1, np.nan)
d["kids"] = d.kid_category_desc.astype("category").cat.codes.replace(-1, np.nan)
d["has_demo"] = 1.0
dcols = ["classification_1_num","classification_2","classification_3_num","classification_4_num","classification_5_num","homeowner","kids","has_demo"]
base = base.merge(d[["household_key"]+dcols], on="household_key", how="left")
base["has_demo"] = base["has_demo"].fillna(0.0)
base = pd.get_dummies(base, columns=["classification_2"], dummy_na=True)
dnum = ["classification_1_num","classification_3_num","classification_4_num","classification_5_num","homeowner","kids","has_demo","classification_2_X","classification_2_Y","classification_2_Z"]

def prep(df, cols, tr, va, logt=False):
    Xtr=df.loc[tr, cols].astype(float).values; ytr=df.loc[tr].future_spend_4w.values
    Xva=df.loc[va, cols].astype(float).values; yva=df.loc[va].future_spend_4w.values
    med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    return np.c_[np.ones(len(Xtr)),Xtr], ytr, np.c_[np.ones(len(Xva)),Xva], yva

def ridge2(df, cols, tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431], alphas=(3,10,30,100,300,1000,3000), logt=False):
    tr=df.snapshot_day.isin(tr_snaps); va=df.snapshot_day.isin(va_snaps)
    cols=[c for c in cols if c in df.columns]
    Xtr,ytr,Xva,yva = prep(df, cols, tr, va)
    best=(1e9,None)
    yt = np.log1p(ytr) if logt else ytr
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@yt)
        p = np.expm1(Xva@w) if logt else Xva@w
        p=np.clip(p,0,None)
        m=np.mean(np.abs(p-yva))
        if m<best[0]: best=(m,a)
    return best

print("E003 ref:", ridge2(base, feat))
print("E003 log-target:", ridge2(base, feat, logt=True))
# interaction block: key spend levels x demographics
inter=[]
for s in ["spend_28","spend_84","spend_364_per_28w","x_exp4w","basket_val_28","recency"]:
    for dd in dnum:
        base[f"{s}__{dd}"] = base[s]*base[dd].fillna(0)
        inter.append(f"{s}__{dd}")
print("E003+interactions:", ridge2(base, feat+inter))
print("E003+inter(sub):", ridge2(base, feat+[c for c in inter if "has_demo" in c or "classification_4" in c]))
# calendar interactions
base["wk"]=(base.snapshot_day.astype(int)+8)//7
for nm, ph in [("s1",2*np.pi*(base.wk%52)/52),("c1",2*np.pi*(base.wk%52)/52)]:
    pass
base["cal_s"]=np.sin(2*np.pi*(base.wk%52)/52); base["cal_c"]=np.cos(2*np.pi*(base.wk%52)/52)
ci=[f"{s}__cal" for s in ["spend_28","spend_84","x_exp4w"]]
for s in ["spend_28","spend_84","x_exp4w"]:
    base[f"{s}__cal"]=base[s]*base["cal_s"]
print("E003+cal-inter:", ridge2(base, feat+ci))
print("E003+demo+inter:", ridge2(base, feat+dnum+inter))
# per-snapshot demeaning check: does a snapshot-level intercept shift help?
print("spend_28 alone:", ridge2(base, ["spend_28"]))
print("spend_84 alone:", ridge2(base, ["spend_84"]))
print("x_exp4w alone:", ridge2(base, ["x_exp4w"]))
print("spend_28+spend_84+x_exp4w:", ridge2(base, ["spend_28","spend_84","x_exp4w"]))