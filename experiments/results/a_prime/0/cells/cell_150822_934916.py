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