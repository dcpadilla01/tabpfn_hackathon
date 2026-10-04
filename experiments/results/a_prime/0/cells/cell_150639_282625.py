import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

# demographics (static) + calendar (function of snapshot_day only)
dem = agent_api.snapshot().demographics
print("dem households:", dem.household_key.nunique())
d = dem.copy()
for c in ["classification_1","classification_3","classification_4","classification_5"]:
    d[c+"_num"] = d[c].astype(str).str.extract(r"(\d+)").astype(float)
d["homeowner"] = d.homeowner_desc.astype("category").cat.codes.replace(-1, np.nan)
d["kids"] = d.kid_category_desc.astype("category").cat.codes.replace(-1, np.nan)
d["has_demo"] = 1.0
dcols = ["household_key","classification_1_num","classification_2","classification_3_num","classification_4_num","classification_5_num","homeowner","kids","has_demo"]
base_d = base.merge(d[dcols], on="household_key", how="left")
base_d["has_demo"] = base_d["has_demo"].fillna(0.0)
base_d["snap_day"] = base_d.snapshot_day.astype(float)
wk = ((base_d.snapshot_day.astype(int)+8)//7)
base_d["wk_sin"] = np.sin(2*np.pi*wk/52.0); base_d["wk_cos"] = np.cos(2*np.pi*wk/52.0)
base_d["wk_idx"] = wk % 52
dcols2 = dcols[1:]+["snap_day","wk_sin","wk_cos","wk_idx"]
base_d = pd.get_dummies(base_d, columns=["classification_2"], dummy_na=True)

def ridge_eval(df, cols, tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431], alphas=(1,3,10,30,100,300), ret_pred=False):
    tr=df[df.snapshot_day.isin(tr_snaps)]; va=df[df.snapshot_day.isin(va_snaps)]
    cols=[c for c in cols if c in df.columns]
    Xtr=tr[cols].astype(float).values; ytr=tr.future_spend_4w.values
    Xva=va[cols].astype(float).values; yva=va.future_spend_4w.values
    med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None,None)
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@ytr); p=np.clip(Xva@w,0,None)
        m=np.mean(np.abs(p-yva))
        if m<best[0]: best=(m,a,p)
    return (best[0],best[1],best[2]) if ret_pred else (best[0],best[1])

print("E003 ref:", ridge_eval(base_d, feat))
print("E003+demo:", ridge_eval(base_d, feat+dcols2))
print("E003+cal:", ridge_eval(base_d, feat+["snap_day","wk_sin","wk_cos","wk_idx"]))
print("E003+demo+cal:", ridge_eval(base_d, feat+dcols2+["snap_day","wk_sin","wk_cos","wk_idx"]))

m,a,p = ridge_eval(base_d, feat, ret_pred=True)
va = base_d[base_d.snapshot_day.isin([375,403,431])]
y = va.future_spend_4w.values
print("per-snap MAE:", {s: round(np.mean(np.abs(p[va.snapshot_day==s]-y[va.snapshot_day==s])),2) for s in [375,403,431]})
qs = np.quantile(y, [0,.2,.4,.6,.8,.95,1.0])
for i in range(len(qs)-1):
    msk = (y>=qs[i]) & (y<=qs[i+1])
    print(f"y in [{qs[i]:.0f},{qs[i+1]:.0f}] n={msk.sum()} MAE={np.mean(np.abs(p[msk]-y[msk])):.1f}")
print("naive spend_28 MAE:", np.mean(np.abs(va.spend_28.values-y)))
hh_mean = va.groupby("household_key").future_spend_4w.mean()
yy = va.set_index("household_key").future_spend_4w
print("oracle hh-mean MAE:", np.mean(np.abs(hh_mean.reindex(yy.index).values-yy.values)))
print("E003 pred mean vs y mean:", p.mean(), y.mean())