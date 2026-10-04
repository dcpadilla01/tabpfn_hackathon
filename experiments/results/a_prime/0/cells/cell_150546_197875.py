import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
season = agent_api.load_saved("season_v1.parquet").drop(columns=["index"], errors="ignore")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]
sf=[c for c in season.columns if c not in ("household_key","snapshot_day")]
base_se = base.merge(season, on=["household_key","snapshot_day"], how="left")
print("season corr with y:", {c: round(np.corrcoef(base_se[c].astype(float), base_se.future_spend_4w)[0,1],3) for c in sf})

def ridge_eval(df, cols, tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431], alphas=(1,3,10,30,100,300)):
    tr=df[df.snapshot_day.isin(tr_snaps)]; va=df[df.snapshot_day.isin(va_snaps)]
    Xtr=tr[cols].astype(float).values; ytr=tr.future_spend_4w.values
    Xva=va[cols].astype(float).values; yva=va.future_spend_4w.values
    med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None)
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@ytr); p=np.clip(Xva@w,0,None)
        m=np.mean(np.abs(p-yva))
        if m<best[0]: best=(m,a)
    return best

print("E003 ref:", ridge_eval(base, feat))
print("E003+season:", ridge_eval(base_se, feat+sf))
struct = agent_api.load_saved("structure_v1.parquet").drop(columns=["index"], errors="ignore")
stf=[c for c in struct.columns if c not in ("household_key","snapshot_day")]
base_ss = base_se.merge(struct, on=["household_key","snapshot_day"], how="left")
print("E003+season+struct:", ridge_eval(base_ss, feat+sf+stf))