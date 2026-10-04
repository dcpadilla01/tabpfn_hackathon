import pandas as pd
for name in ["mkt_v2","hist_v2","season_v1","mix_v1","e006_temporal","e007_robust","structure_v1","e005_full_plus_mix"]:
    try:
        df = agent_api.load_saved(name+".parquet")
        print(name, df.shape)
        print(list(df.columns))
        print("---")
    except Exception as e:
        print(name, "ERR", e)

tt = agent_api.train_targets()
print(tt.shape, tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w==0).mean())
print(agent_api.snapshot_days())


# ---- cell ----

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


# ---- cell ----

import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
struct = agent_api.load_saved("structure_v1.parquet")
mix = agent_api.load_saved("mix_v1.parquet")
rob = agent_api.load_saved("e007_robust.parquet")
tmp = agent_api.load_saved("e006_temporal.parquet")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
def add(df, pfx):
    f = df.drop(columns=["index"], errors="ignore")
    f = f.drop(columns=[c for c in mkt.columns if c in f.columns and c not in ("household_key","snapshot_day")])
    return base.merge(f, on=["household_key","snapshot_day"], how="left")
base_s = add(struct, None); base_m = add(mix, None); base_r = add(rob, None); base_t = add(tmp, None)
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(df, cols, tr_snaps, va_snaps, alphas=(0.3,1,3,10,30,100,300)):
    tr = df[df.snapshot_day.isin(tr_snaps)]; va = df[df.snapshot_day.isin(va_snaps)]
    Xtr = tr[cols].astype(float).values; ytr = tr.future_spend_4w.values
    Xva = va[cols].astype(float).values; yva = va.future_spend_4w.values
    med = np.nanmedian(Xtr, axis=0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None)
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@ytr); p=np.clip(Xva@w,0,None)
        mae=np.mean(np.abs(p-yva))
        if mae<best[0]: best=(mae,a)
    return best

tr_s=[95,123,151,179,207,235,263,291,319,347]; va_s=[375,403,431]
sf=[c for c in struct.columns if c not in ("household_key","snapshot_day","index")]
print("E003+struct:", ridge_eval(base_s, feat+sf, tr_s, va_s))
print("struct only:", ridge_eval(base_s, sf, tr_s, va_s))
mf=[c for c in mix.columns if c not in ("household_key","snapshot_day")]
print("E003+mix:", ridge_eval(base_m, feat+mf, tr_s, va_s))
rf=[c for c in rob.columns if c not in ("household_key","snapshot_day") and c not in feat]
print("E003+robust:", ridge_eval(base_r, feat+rf, tr_s, va_s))
tf=[c for c in tmp.columns if c not in ("household_key","snapshot_day") and c not in feat]
print("E003+temporal:", ridge_eval(base_t, feat+tf, tr_s, va_s))
y=base.future_spend_4w.values; cors={}
for c in feat:
    x=base[c].astype(float).values; ok=~np.isnan(x)
    if ok.sum()>100: cors[c]=abs(np.corrcoef(x[ok],y[ok])[0,1])
top=sorted(cors,key=cors.get,reverse=True)
print("top15 corr:", [(c,round(cors[c],3)) for c in top[:15]])
for k in (20,30,40,50,60):
    print(f"E003 top{k}corr:", ridge_eval(base, top[:k], tr_s, va_s))


# ---- cell ----
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

# ---- cell ----
import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
season = agent_api.load_saved("season_v1.parquet").drop(columns=["index"], errors="ignore")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]
sf=[c for c in season.columns if c not in ("household_key","snapshot_day") and c!="tenure"]
base_se = base.merge(season.drop(columns=["tenure"]), on=["household_key","snapshot_day"], how="left")
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

# ---- cell ----
import pandas as pd, numpy as np
s = agent_api.load_saved("season_v1.parquet")
print(s.dtypes)
print(s.head(3))
print(s.snapshot_day.value_counts().head(20))
tt = agent_api.train_targets()
print(tt.dtypes)
print(tt.head(3))
print("tt snaps:", sorted(tt.snapshot_day.unique()))
mkt = agent_api.load_saved("mkt_v2.parquet")
print(mkt.dtypes.head(3)); print(mkt.head(2))

# ---- cell ----
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

# ---- cell ----
import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

print("mean target by snapshot:")
print(tt.groupby("snapshot_day").future_spend_4w.agg(["mean","median"]).round(1).T)
v = agent_api.snapshot(459)
t = v.transactions
dspend = t.groupby("day").sales_value.sum()
blk = dspend.groupby((dspend.index-1)//28).sum()
print("28d-block aggregate spend (k$: {block: spend}):", {int(k): round(x/1000,1) for k,x in blk.items()})

def ridge_eval(df, cols, tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431], alphas=(1,3,10,30,100,300)):
    tr=df[df.snapshot_day.isin(tr_snaps)]; va=df[df.snapshot_day.isin(va_snaps)]
    cols=[c for c in cols if c in df.columns]
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

def blocks(view, s):
    t = view.transactions
    hh = view.households
    g = t.groupby("household_key")
    out = hh.set_index("household_key")[[]].copy()
    m13 = (t.day > s-364) & (t.day <= s-336)
    p13 = t[m13].groupby("household_key").sales_value.sum()
    m26 = (t.day > s-392) & (t.day <= s-364)
    p13b = t[m26].groupby("household_key").sales_value.sum()
    out["p13"] = p13; out["p13_avg2"] = (p13+p13b)/2
    s28 = t[t.day>s-28].groupby("household_key").sales_value.sum()
    out["p13_div_s28"] = p13/s28.replace(0,np.nan)
    out["p13_zero"] = (p13<=0).astype(float)
    age = (s - t.day).astype(float)
    for hl in (14, 28, 56, 112):
        w = np.power(0.5, age/hl)
        out[f"ew{hl}"] = (t.sales_value*w).groupby(t.household_key).sum()/w.groupby(t.household_key).sum()
    s84 = t[t.day>s-84].groupby("household_key").sales_value.sum()
    s364 = t[t.day>s-364].groupby("household_key").sales_value.sum()
    for k in (2,5,10):
        out[f"blend{k}"] = (s28 + k*s84/3.0)/(1.0+k/3.0)
        out[f"blend364_{k}"] = (s28 + k*s364/13.0)/(1.0+k/13.0)
    t84 = t[t.day>s-84]
    bs = t84.groupby("basket_id").agg(sv=("sales_value","sum"), store=("store_id","first"))
    hh_store = t84.groupby("household_key").store_id.agg(lambda x: x.value_counts().idxmax())
    store_mean = bs.groupby("store").sv.mean()
    out["store_mean_bv"] = hh_store.map(store_mean)
    out["n_stores84b"] = t84.groupby("household_key").store_id.nunique()
    ph = 2*np.pi*(s % 364)/364.0
    out["cal_sin"] = np.sin(ph); out["cal_cos"] = np.cos(ph)
    out["cal_sin2"] = np.sin(2*ph); out["cal_cos2"] = np.cos(2*ph)
    return out

B = agent_api.build_features(blocks)
print("blocks built:", B.shape, list(B.columns))
agent_api.save_table(B.reset_index(), "blocks_v1.parquet")
base_b = base.merge(B.reset_index(), on=["household_key","snapshot_day"], how="left")
bl = list(B.columns)
print("null rates:", {c: round(base_b[c].isna().mean(),3) for c in bl})
print("REF E003:", ridge_eval(base_b, feat))
for grp,name in [([c for c in bl if c.startswith('p13')],"p13"),
                 ([c for c in bl if c.startswith('ew')],"ewma"),
                 ([c for c in bl if c.startswith('blend')],"blend"),
                 (["store_mean_bv","n_stores84b"],"store"),
                 ([c for c in bl if c.startswith('cal')],"cal")]:
    print(f"E003+{name}:", ridge_eval(base_b, feat+grp))
print("E003+p13+ewma:", ridge_eval(base_b, feat+[c for c in bl if c.startswith('p13') or c.startswith('ew')]))
print("E003+p13+ewma+blend:", ridge_eval(base_b, feat+[c for c in bl if c.startswith('p13') or c.startswith('ew') or c.startswith('blend')]))
print("E003+ALL:", ridge_eval(base_b, feat+bl))

# ---- cell ----
import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
base = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

def ridge_eval(df, cols, tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431], alphas=(1,3,10,30,100,300)):
    tr=df[df.snapshot_day.isin(tr_snaps)]; va=df[df.snapshot_day.isin(va_snaps)]
    cols=[c for c in cols if c in df.columns]
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

def blocks(view, s):
    t = view.transactions
    hh = view.households
    out = pd.DataFrame(index=hh)
    m13 = (t.day > s-364) & (t.day <= s-336)
    p13 = t[m13].groupby("household_key").sales_value.sum()
    m26 = (t.day > s-392) & (t.day <= s-364)
    p13b = t[m26].groupby("household_key").sales_value.sum()
    out["p13"] = p13; out["p13_avg2"] = (p13+p13b)/2
    s28 = t[t.day>s-28].groupby("household_key").sales_value.sum()
    out["p13_div_s28"] = p13/s28.replace(0,np.nan)
    out["p13_zero"] = (p13<=0).astype(float)
    age = (s - t.day).astype(float)
    for hl in (14, 28, 56, 112):
        w = np.power(0.5, age/hl)
        out[f"ew{hl}"] = (t.sales_value*w).groupby(t.household_key).sum()/w.groupby(t.household_key).sum()
    s84 = t[t.day>s-84].groupby("household_key").sales_value.sum()
    s364 = t[t.day>s-364].groupby("household_key").sales_value.sum()
    for k in (2,5,10):
        out[f"blend{k}"] = (s28 + k*s84/3.0)/(1.0+k/3.0)
        out[f"blend364_{k}"] = (s28 + k*s364/13.0)/(1.0+k/13.0)
    t84 = t[t.day>s-84]
    bs = t84.groupby("basket_id").agg(sv=("sales_value","sum"), store=("store_id","first"))
    hh_store = t84.groupby("household_key").store_id.agg(lambda x: x.value_counts().idxmax())
    store_mean = bs.groupby("store").sv.mean()
    out["store_mean_bv"] = hh_store.map(store_mean)
    out["n_stores84b"] = t84.groupby("household_key").store_id.nunique()
    ph = 2*np.pi*(s % 364)/364.0
    out["cal_sin"] = np.sin(ph); out["cal_cos"] = np.cos(ph)
    out["cal_sin2"] = np.sin(2*ph); out["cal_cos2"] = np.cos(2*ph)
    return out

B = agent_api.build_features(blocks)
print("blocks built:", B.shape, list(B.columns))
agent_api.save_table(B.reset_index(), "blocks_v1.parquet")
base_b = base.merge(B.reset_index(), on=["household_key","snapshot_day"], how="left")
bl = list(B.columns)
print("null rates:", {c: round(base_b[c].isna().mean(),3) for c in bl})
print("REF E003:", ridge_eval(base_b, feat))
for grp,name in [([c for c in bl if c.startswith('p13')],"p13"),
                 ([c for c in bl if c.startswith('ew')],"ewma"),
                 ([c for c in bl if c.startswith('blend')],"blend"),
                 (["store_mean_bv","n_stores84b"],"store"),
                 ([c for c in bl if c.startswith('cal')],"cal")]:
    print(f"E003+{name}:", ridge_eval(base_b, feat+grp))
print("E003+p13+ewma:", ridge_eval(base_b, feat+[c for c in bl if c.startswith('p13') or c.startswith('ew')]))
print("E003+p13+ewma+blend:", ridge_eval(base_b, feat+[c for c in bl if c.startswith('p13') or c.startswith('ew') or c.startswith('blend')]))
print("E003+ALL:", ridge_eval(base_b, feat+bl))

# ---- cell ----
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

# ---- cell ----
import numpy as np, pandas as pd
tt = agent_api.train_targets()
mkt = agent_api.load_saved("mkt_v2.parquet")
feat=[c for c in mkt.columns if c not in ("household_key","snapshot_day")]

def dist_block(view, s):
    t = view.transactions
    hh = view.households
    out = pd.DataFrame(index=hh)
    first = t.groupby("household_key").day.min()
    ten = (s - first).astype(float)
    nb = np.floor(ten/28).astype(int).clip(lower=0, upper=13)
    # aligned prior blocks: k=1..13 window (s-28k, s-28k+28]
    cols = {}
    for k in range(1, 14):
        lo, hi = s-28*k, s-28*k+28
        cols[k] = t[(t.day > lo) & (t.day <= hi)].groupby("household_key").sales_value.sum()
    M = pd.DataFrame(cols)
    avail = {}
    for h in M.index:
        n = int(nb.get(h, 0))
        avail[h] = M.loc[h, [k for k in range(1,14) if k <= n]].values.astype(float) if n>0 else np.array([])
    def stat(fn):
        return pd.Series({h: fn(v) for h, v in avail.items()})
    out["d_nblk"] = pd.Series({h: len(v) for h,v in avail.items()})
    out["d_med"] = stat(lambda v: np.median(v) if len(v) else np.nan)
    out["d_mean"] = stat(lambda v: np.mean(v) if len(v) else np.nan)
    out["d_std"] = stat(lambda v: np.std(v) if len(v) else np.nan)
    out["d_iqr"] = stat(lambda v: (np.percentile(v,75)-np.percentile(v,25)) if len(v) else np.nan)
    out["d_zerofrac"] = stat(lambda v: np.mean(v<=1.0) if len(v) else np.nan)
    out["d_max"] = stat(lambda v: np.max(v) if len(v) else np.nan)
    out["d_min"] = stat(lambda v: np.min(v) if len(v) else np.nan)
    s28 = t[t.day>s-28].groupby("household_key").sales_value.sum()
    out["d_cur_over_med"] = s28/out["d_med"]
    out["d_last_zero"] = (s28<=1.0).astype(float)
    return out

D = agent_api.build_features(dist_block)
print("dist built:", D.shape, list(D.columns))
dl = [c for c in D.columns if c.startswith("d_")]
Dfull = mkt.merge(D.reset_index(), on=["household_key","snapshot_day"], how="inner")
print("null rates:", {c: round(Dfull[c].isna().mean(),3) for c in dl})
agent_api.save_table(Dfull, "e008_dist.parquet")

def prep_cols(df, cols, tr, va):
    Xtr=df.loc[tr, cols].astype(float).values; ytr=df.loc[tr].future_spend_4w.values
    Xva=df.loc[va, cols].astype(float).values; yva=df.loc[va].future_spend_4w.values
    med=np.nanmedian(Xtr,0); med=np.where(np.isnan(med),0,med)
    Xtr=np.where(np.isnan(Xtr),med,Xtr); Xva=np.where(np.isnan(Xva),med,Xva)
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
    return (Xtr-mu)/sd, ytr, (Xva-mu)/sd, yva

def ridge2(df, cols, alphas=(10,100,300,1000,3000,10000), tr_snaps=[95,123,151,179,207,235,263,291,319,347], va_snaps=[375,403,431]):
    tr=df.snapshot_day.isin(tr_snaps); va=df.snapshot_day.isin(va_snaps)
    cols=[c for c in cols if c in df.columns]
    Xtr,ytr,Xva,yva = prep_cols(df, cols, tr, va)
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xva=np.c_[np.ones(len(Xva)),Xva]
    best=(1e9,None)
    for a in alphas:
        A=Xtr.T@Xtr+a*np.eye(Xtr.shape[1]); A[-1,-1]-=a
        w=np.linalg.solve(A,Xtr.T@ytr); p=np.clip(Xva@w,0,None)
        m=np.mean(np.abs(p-yva))
        if m<best[0]: best=(m,a)
    return best

# calibration vs harness: harness order E003(63.32) < E006(63.38) < E005(63.41) < E007(63.44)
rob = agent_api.load_saved("e007_robust.parquet"); tmp = agent_api.load_saved("e006_temporal.parquet")
mix = agent_api.load_saved("mix_v1.parquet")
rf=[c for c in rob.columns if c not in ("household_key","snapshot_day") and c not in feat]
tf=[c for c in tmp.columns if c not in ("household_key","snapshot_day") and c not in feat]
mf=[c for c in mix.columns if c not in ("household_key","snapshot_day")]
ttm = tt.merge(mkt, on=["household_key","snapshot_day"], how="inner")
print("CALIB E003:", ridge2(ttm, feat))
print("CALIB E005(E003+mix):", ridge2(ttm.merge(mix,on=["household_key","snapshot_day"],how="left"), feat+mf))
print("CALIB E006(+temporal):", ridge2(ttm.merge(tmp,on=["household_key","snapshot_day"],how="left"), feat+tf))
print("CALIB E007(+robust):", ridge2(ttm.merge(rob,on=["household_key","snapshot_day"],how="left"), feat+rf))
Dm = ttm.merge(D.reset_index(), on=["household_key","snapshot_day"], how="left")
print("RIDGE E003+dist:", ridge2(Dm, feat+dl))

# tiny hand-rolled GBM proxy (depth-3, squared loss)
def gbm_eval(df, cols, rounds=80, lr=0.15, max_depth=3, min_leaf=60):
    try:
        tr=df.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347]); va=df.snapshot_day.isin([375,403,431])
        cols=[c for c in cols if c in df.columns]
        Xtr,ytr,Xva,yva = prep_cols(df, cols, tr, va)
        n,d = Xtr.shape
        pred = np.full(n, ytr.mean()); pva = np.full(len(Xva), ytr.mean())
        rng = np.random.RandomState(0)
        for r in range(rounds):
            g = ytr - pred
            trees = []
            idx_nodes = [np.arange(n)]; pv_nodes=[np.arange(len(Xva))]
            # grow tree level by level
            node_assign = np.zeros(n, dtype=int); node_assign_va = np.zeros(len(Xva), dtype=int)
            leaves = {0: np.arange(n)}; leaves_va = {0: np.arange(len(Xva))}
            leaf_val = {}
            for depth in range(max_depth):
                newleaves={}; newleaves_va={}; newleaf_val={}
                for nid, idx in leaves.items():
                    Xn, gn = Xtr[idx], g[idx]
                    if len(idx) < 2*min_leaf:
                        newleaves[nid]=idx; newleaves_va[nid]=leaves_va[nid]; newleaf_val[nid]=gn.mean(); continue
                    best=( -1e18, None, None)
                    for j in range(d):
                        xs = Xn[:,j]
                        o = np.argsort(xs, kind="stable"); xs_s=xs[o]; gs=gn[o]
                        cs=np.cumsum(gs); tot=cs[-1]; cnt=len(idx)
                        cl=cs; cr=tot-cs; nl=np.arange(1,cnt+1); nr=cnt-nl
                        valid=(xs_s[1:]>xs_s[:-1]) & (nl>=min_leaf) & (nr>=min_leaf)
                        if not valid.any(): continue
                        gain = cl[:-1]**2/(nl[:-1]+1e-9) + cr[:-1]**2/(nr[:-1]+1e-9) - tot**2/(cnt+1e-9)
                        gain = np.where(valid, gain, -1e18)
                        k = int(np.argmax(gain))
                        if gain[k] > best[0]: best=(gain[k], j, (xs_s[k]+xs_s[k+1])/2)
                    if best[1] is None:
                        newleaves[nid]=idx; newleaves_va[nid]=leaves_va[nid]; newleaf_val[nid]=gn.mean(); continue
                    j, thr = best[1], best[2]
                    msk = Xn[:,j] <= thr
                    li = idx[msk]; ri = idx[~msk]
                    lva = leaves_va[nid][Xva[leaves_va[nid],j] <= thr]; rva = leaves_va[nid][Xva[leaves_va[nid],j] > thr]
                    newleaves[2*nid+1]=li; newleaves[2*nid+2]=ri
                    newleaves_va[2*nid+1]=lva; newleaves_va[2*nid+2]=rva
                    newleaf_val[2*nid+1]=g[li].mean(); newleaf_val[2*nid+2]=g[ri].mean()
                leaves, leaves_va, leaf_val = newleaves, newleaves_va, newleaf_val
            upd = np.zeros(n); upd_va = np.zeros(len(Xva))
            for nid, idx in leaves.items():
                upd[idx] = leaf_val[nid]
                upd_va[leaves_va[nid]] = leaf_val[nid]
            pred += lr*upd; pva += lr*upd_va
        pva = np.clip(pva, 0, None)
        return round(float(np.mean(np.abs(pva-yva))),3)
    except Exception as e:
        return f"GBM proxy failed: {e}"
print("GBM E003:", gbm_eval(ttm, feat))
print("GBM E003+dist:", gbm_eval(Dm, feat+dl))