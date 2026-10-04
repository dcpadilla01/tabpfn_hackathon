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
Xdf = m[fcols].copy()
for c in fcols:
    if Xdf[c].dtype == object: Xdf[c] = pd.factorize(Xdf[c])[0].astype(float)
    Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
Xdf = Xdf.replace([np.inf,-np.inf], np.nan)
X = Xdf.values; y = m.future_spend_4w.values; days = m.snapshot_day.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)
Z1 = np.hstack([Z, np.ones((len(Z),1))])
K = Z1.shape[1]

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
sel_prev = [c for c in sel_prev if c in fcols]
SP = [fcols.index(c) for c in sel_prev]

def fit_pred(sub, tr_mask, va_mask, lam=30.0):
    G = Z1[tr_mask][:,sub].T@Z1[tr_mask][:,sub]; b = Z1[tr_mask][:,sub].T@y[tr_mask]
    G += lam*np.eye(len(sub)); G[-1,-1] -= lam
    w = np.linalg.solve(G, b)
    return np.clip(Z1[va_mask][:,sub]@w,0,None)

def loo(sub, lam=30.0):
    errs=[]
    for d in TRAIN:
        p = fit_pred(sub, days!=d, days==d, lam)
        errs.append(np.abs(p-y[days==d]))
    return float(np.concatenate(errs).mean())

def extrap(sub, lam=30.0):
    """train <=375, validate 403+431"""
    tr = days<=375; va = (days==403)|(days==431)
    p = fit_pred(sub, tr, va, lam)
    return float(np.abs(p-y[va]).mean())

# candidate new features
def addf(name, vec):
    global Z1, fcols
    v = np.asarray(vec, float)
    v = (v-np.nanmean(v))/ (np.nanstd(v)+1e-12)
    v = np.clip(np.where(np.isnan(v),0,v),-8,8)
    Z1 = np.hstack([Z1, v.reshape(-1,1)])
    fcols = fcols+[name]
    return len(fcols)-1

i_zw13 = addf("x_zerow13", X[:, fcols.index("r_zerow13")])
i_zw4  = addf("x_zerow4",  X[:, fcols.index("r_zerow4")])
i_dsl  = addf("x_dsl",     X[:, fcols.index("r_dsl")])
i_zm13 = addf("x_zerom13", X[:, fcols.index("r_zerom13")])
i_sday = addf("x_sday",    days.astype(float))
i_gml4 = addf("x_gml4",    X[:, fcols.index("g_mean_l4")])
top = X[:, fcols.index("nf_pow90_ewm4")]
topz = np.clip(np.where(np.isnan(top),0,(top-np.nanmean(top))/(np.nanstd(top)+1e-12)),-8,8)
i_sx   = addf("x_sday_x_top", days.astype(float)/100*topz)
i_gxt  = addf("x_gml4_x_top", X[:, fcols.index("g_mean_l4")]/100*topz)
# conditional level: spend_l1 per active day
cl = X[:, fcols.index("spend_l1")]/(X[:, fcols.index("days_active_l1")].clip(1,None))
i_cl  = addf("x_cond_level", cl)
# decile dummies of top feature
qs = np.nanquantile(top, np.linspace(0.1,0.9,9))
binned = np.searchsorted(qs, np.nan_to_num(top, nan=-1e9))
for b in range(1,9):
    addf(f"x_bin{b}", (binned==b).astype(float))

names = {i_zw13:"x_zerow13",i_zw4:"x_zerow4",i_dsl:"x_dsl",i_zm13:"x_zerom13",i_sday:"x_sday",i_gml4:"x_gml4",i_sx:"x_sday_x_top",i_gxt:"x_gml4_x_top",i_cl:"x_cond_level"}
print(f"{'variant':34s} {'LOO':>7s} {'EXTR':>7s}")
base_loo, base_ex = loo(SP), extrap(SP)
print(f"{'prev18':34s} {base_loo:7.3f} {base_ex:7.3f}")
tests = {
 "prev18+zerow13": SP+[i_zw13],
 "prev18+zerow13+zerow4+dsl": SP+[i_zw13,i_zw4,i_dsl],
 "prev18+zerow13+zerow4+dsl+zm13": SP+[i_zw13,i_zw4,i_dsl,i_zm13],
 "prev18+sday": SP+[i_sday],
 "prev18+gml4": SP+[i_gml4],
 "prev18+sday+gml4": SP+[i_sday,i_gml4],
 "prev18+sday_x_top": SP+[i_sx],
 "prev18+gml4_x_top": SP+[i_gxt],
 "prev18+cond_level": SP+[i_cl],
 "prev18+zerow..+sday+gml4": SP+[i_zw13,i_zw4,i_dsl,i_sday,i_gml4],
 "prev18+zerow..+cond_level": SP+[i_zw13,i_zw4,i_dsl,i_cl],
 "prev18+allbins": SP+[j for j in range(K, len(fcols)) if fcols[j].startswith("x_bin")],
 "prev18+zerow13+allbins": SP+[i_zw13]+[j for j in range(K, len(fcols)) if fcols[j].startswith("x_bin")],
}
for nm, sub in tests.items():
    print(f"{nm:34s} {loo(sub):7.3f} {extrap(sub):7.3f}")
print("\nnote: EXTR = train<=375, val={403,431}; LOO = leave-one-snapshot-out")