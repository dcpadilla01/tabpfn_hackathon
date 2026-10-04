
import numpy as np, pandas as pd

tt = agent_api.train_targets()
def load(n): 
    df = agent_api.load_saved(n+".parquet")
    return df

mkt = load("mkt_v2")            # E003 features (74 cols)
struct = load("structure_v1").drop(columns=["index"], errors="ignore")
mix = load("mix_v1")
rob = load("e007_robust")
tmp = load("e006_temporal")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
print("merged", base.shape)

def ridge_eval(cols, tr_snaps, va_snaps, alphas=(0.3,1,3,10,30,100,300), clip=True, label=""):
    tr = base[base.snapshot_day.isin(tr_snaps)]
    va = base[base.snapshot_day.isin(va_snaps)]
    Xtr = tr[cols].astype(float).values; ytr = tr.future_spend_4w.values
    Xva = va[cols].astype(float).values; yva = va.future_spend_4w.values
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isnan(med),0,med)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None)
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = Xva@w
        if clip: p=np.clip(p,0,None)
        mae = np.mean(np.abs(p-yva))
        if mae<best[0]: best=(mae,a)
    return best

tr_s=[95,123,151,179,207,235,263,291,319,347]; va_s=[375,403,431]
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]
print("spend_28 only:", ridge_eval(["spend_28"],tr_s,va_s,label="s28"))
print("E001-ish (17):", ridge_eval(feat[:15],tr_s,va_s))
print("E003 all:", ridge_eval(feat,tr_s,va_s))
sf=[c for c in struct.columns if c not in ("household_key","snapshot_day")]
print("E003+struct:", ridge_eval(feat+sf,tr_s,va_s))
mf=[c for c in mix.columns if c not in ("household_key","snapshot_day")]
print("E003+mix:", ridge_eval(feat+mf,tr_s,va_s))
rf=[c for c in rob.columns if c not in ("household_key","snapshot_day") and c not in feat]
print("E003+robust:", ridge_eval(feat+rf,tr_s,va_s))
tf=[c for c in tmp.columns if c not in ("household_key","snapshot_day") and c not in feat]
print("E003+temporal:", ridge_eval(feat+tf,tr_s,va_s))
# correlation screen: top-k |corr| with y among E003 feats
y=base.future_spend_4w.values
cors={}
for c in feat:
    x=base[c].astype(float).values
    ok=~np.isnan(x)
    if ok.sum()>100: cors[c]=abs(np.corrcoef(x[ok],y[ok])[0,1])
top=sorted(cors,key=cors.get,reverse=True)
print("top20 corr:", [(c,round(cors[c],3)) for c in top[:20]])
for k in (20,30,40,50,60):
    print(f"E003 top{k}corr:", ridge_eval(top[:k],tr_s,va_s))
