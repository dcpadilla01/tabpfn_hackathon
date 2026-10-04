import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
key = ["household_key","snapshot_day"]
pool = A.load_saved("e003_catmix.parquet").copy()
for nm in ["nf_p1","e004_mkt_full","e002_channel","nf_transforms","nf_candidates","nf_robust"]:
    df = A.load_saved(nm+".parquet")
    newc = [c for c in df.columns if c not in pool.columns and c not in key]
    pool = pool.merge(df[key+newc], on=key, how="left")
fcols = [c for c in pool.columns if c not in key]
m = tt.merge(pool, on=key, how="left")

# --- build new candidate features on the FULL train rows (approx; ok for screening) ---
v = A.snapshot()
tx = v.transactions
demo = v.demographics
# store tier: mean sales per basket per store (up to day 459)
sb = tx.groupby(["store_id","basket_id"]).sales_value.sum().reset_index()
store_tier = sb.groupby("store_id").sales_value.mean()
store_n = sb.groupby("store_id").size()
# household main store over last 84d
hh_days = [95,123,151,179,207,235,263,291,319,347,375,403,431,459]
rows = []
for s in [95,123,151,179,207,235,263,291,319,347,375,403,431]:
    t = tx[(tx.day<=s)&(tx.day>s-84)]
    b = t.groupby(["household_key","basket_id","store_id"]).sales_value.sum().reset_index()
    hs = b.groupby(["household_key","store_id"]).agg(sp=("sales_value","sum")).reset_index()
    tot = hs.groupby("household_key").sp.transform("sum")
    hs["share"] = hs.sp/tot
    main = hs.loc[hs.groupby("household_key").share.idxmax(), ["household_key","store_id","share"]]
    main["snapshot_day"] = s
    rows.append(main)
mainst = pd.concat(rows).rename(columns={"store_id":"main_store","share":"main_share"})
m = m.merge(mainst[["household_key","snapshot_day","main_store","main_share"]], on=key, how="left")
m["store_tier"] = m.main_store.map(store_tier)
m["store_logn"] = m.main_store.map(np.log1p(store_n))
# store-mean shrinkage of recent spend: store-level mean of spend_l1
m["store_mean_l1"] = m.groupby("snapshot_day").spend_l1.transform(lambda x: x.groupby(m.main_store).transform("mean"))
# demographics cell means
m2 = m.merge(demo, on="household_key", how="left")
m2["cell"] = m2.classification_4.fillna("NA")+"|"+m2.homeowner_desc.fillna("NA")+"|"+m2.kid_category_desc.fillna("NA")
m2["cell_mean_l1"] = m2.groupby(["snapshot_day","cell"]).spend_l1.transform("mean")
m2["cell_mean_l4"] = m2.groupby(["snapshot_day","cell"]).spend_l123_mean.transform("mean")
m = m2

newf = ["main_share","store_tier","store_logn","store_mean_l1","cell_mean_l1","cell_mean_l4"]
Xdf = m[fcols+newf].copy()
for c in fcols+newf:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]
Gd, bd, Zv, yv = {}, {}, {}, {}
for d in TRAIN:
    tr = days != d; va = days == d
    Gd[d] = Z1[tr].T@Z1[tr]; bd[d] = Z1[tr].T@y[tr]
    Zv[d] = Z1[va]; yv[d] = y[va]
def loo(sub, lam=30.0):
    errs=[]
    for d in TRAIN:
        s=[*sub,K-1]
        G=Gd[d][np.ix_(s,s)].copy(); b=bd[d][s].copy()
        G+=lam*np.eye(len(s)); G[-1,-1]-=lam
        errs.append(np.abs(np.clip(Zv[d][:,s]@np.linalg.solve(G,b),0,None)-yv[d]))
    return float(np.concatenate(errs).mean())

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
SP = [fcols.index(c) for c in sel_prev if c in fcols]
zw = fcols.index("r_zerow13")
print("base prev18+zw13:", round(loo(SP+[zw]),3))
for c in newf:
    print(f"+{c:14s}", round(loo(SP+[zw, fcols.index(c)]),3))
print("+store_tier+store_mean_l1+cell_mean_l1:", round(loo(SP+[zw, fcols.index("store_tier"), fcols.index("store_mean_l1"), fcols.index("cell_mean_l1")]),3))
# demographics columns themselves
democ = ["classification_1","classification_2","classification_3","classification_4","classification_5","homeowner_desc","kid_category_desc","has_demographics"]
democ = [c for c in democ if c in fcols]
print("demo cols found:", democ)
print("+demo cols:", round(loo(SP+[zw]+[fcols.index(c) for c in democ]),3))