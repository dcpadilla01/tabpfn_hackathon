import pandas as pd, numpy as np, agent_api, warnings, re
warnings.filterwarnings("ignore")
tt = agent_api.train_targets()
fe = agent_api.load_saved("e006_zero_inflation.parquet")
df = fe.merge(tt, on=["household_key","snapshot_day"], how="inner")
ycol="future_spend_4w"
cat = [c for c in df.columns if df[c].dtype.name in ("object","category","bool")]
num = [c for c in df.columns if c not in ("household_key","snapshot_day",ycol) and c not in cat]
X = pd.concat([df[num], pd.get_dummies(df[cat].astype(str), dummy_na=True)], axis=1).astype(float)
X = X.replace([np.inf,-np.inf], np.nan).fillna(X.median())
X = X.loc[:, X.std()>0]
y = df[ycol].values
tr = (df.snapshot_day<=403).values; va=(df.snapshot_day==431).values
Xtr=X.values[tr]; ytr=y[tr]; Xva=X.values[va]; yva=y[va]
mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
Ztr=np.c_[np.ones(tr.sum()),(Xtr-mu)/sd]
lam=300; A=Ztr.T@Ztr+lam*np.eye(Ztr.shape[1]); A[0,0]-=lam
w=np.linalg.solve(A,Ztr.T@ytr)
def mae(Zv):
    return float(np.abs(np.clip(np.c_[np.ones(len(Zv)),Zv]@w,0,None)-yva).mean())
Zva=(Xva-mu)/sd
print("BASE MAE(431):", round(mae(Zva),2))
print("median-baseline MAE:", round(float(np.abs(np.median(ytr)-yva).mean()),2))
i28=list(X.columns).index("spend_28")
print("spend_28-only MAE:", round(mae(Zva*np.where(np.arange(len(X.columns))==i28,1,0)) if False else 0,2))
# proper single-feature model for reference
Z1=np.c_[np.ones(tr.sum()),(Xtr[:,i28]-mu[i28])/sd[i28]]
A1=Z1.T@Z1+lam*np.eye(2); A1[0,0]-=lam
w1=np.linalg.solve(A1,ytr)
p1=np.clip(np.c_[np.ones(va.sum()),(Xva[:,i28]-mu[i28])/sd[i28]]@w1,0,None)
print("spend_28-only MAE:", round(float(np.abs(p1-yva).mean()),2))
# block ablation (zero standardized cols = set to train mean)
blocks={}
for c in X.columns:
    if c.startswith("dept_"): b="dept"
    elif c.startswith("z_"): b="zeroint"
    elif c.startswith("weekend"): b="basket"
    elif re.search("camp|redemp|coupon|n_campaigns",c): b="campaign"
    elif re.search("age_code|^class|homeowner|kids_code|size_code|has_demo",c): b="demo"
    elif c.startswith(("week","trend","ly_")): b="season"
    elif re.search("spend|dec_|avg_weekly|weekly_|^r_",c): b="spend"
    elif re.search("trip|gap|zero|active|recency|days_since",c): b="recency_zero"
    elif re.search("basket|lines|unit_price|qty|national|n_products|n_depts|disc|evening",c): b="basket"
    else: b="misc"
    blocks.setdefault(b,[]).append(c)
res=[]
for b,cols in sorted(blocks.items()):
    Zv2=Zva.copy()
    for c in cols: Zv2[:,X.columns.get_loc(c)]=0.0
    res.append((b,len(cols),round(mae(Zv2),2)))
for b,n,m in res: print(f"drop {b:14s} n={n:3d} MAE={m}")
print()
coefs=pd.Series(np.abs(w[1:]),index=X.columns).sort_values(ascending=False)
print("top-15 |coef|:", coefs.head(15).round(1).to_dict())
