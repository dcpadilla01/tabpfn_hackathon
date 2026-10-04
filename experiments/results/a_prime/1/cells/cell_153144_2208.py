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
hh = m.household_key.values
TRAIN = [95,123,151,179,207,235,263,291,319,347,375,403,431]
mu = np.nanmean(X,0); sd = np.nanstd(X,0); sd[sd==0]=1
Z = np.clip(np.where(np.isnan((X-mu)/sd),0.0,(X-mu)/sd),-8,8)

sel_prev = ['nf_pow90_ewm4','spend_rate28','nspend7','days_active_l1','h_act_pow90','trips_l3','nbask_max84','ntrips7','spend28_DELI','ndow5','spend_l456_mean','nf_ratio123v456','mkt_tgt_TypeA_n','nf_wsin1','nf_wsin2','nf_pow90_ewm13','h_inact_s123','nwmax12']
SP = [fcols.index(c) for c in sel_prev if c in fcols]
i_zw13 = fcols.index("r_zerow13")

codes = pd.factorize(hh)[0]
H = np.zeros((len(hh), codes.max()+1), dtype=np.float32)
H[np.arange(len(hh)), codes] = 1.0
print("hh dummy matrix:", H.shape)

def loo_hh(sub, lam_f=30.0, lam_h=100.0, use_hh=True, inter_zw=False):
    F = np.hstack([Z[:,sub], np.ones((len(Z),1))]).astype(np.float64)
    if inter_zw:
        zw = (Z[:,i_zw13] > 0.5).astype(float)
        F = np.hstack([F, Z[:,sub]*zw[:,None]])
    errs = []
    for d in TRAIN:
        tr = days != d; va = days == d
        Atr = np.hstack([F[tr]] + ([H[tr].astype(np.float64)] if use_hh else []))
        Ava = np.hstack([F[va]] + ([H[va].astype(np.float64)] if use_hh else []))
        lam = np.array([lam_f]*F.shape[1] + ([lam_h]*H.shape[1] if use_hh else []))
        G = Atr.T@Atr + np.diag(lam)
        w = np.linalg.solve(G, Atr.T@y[tr])
        p = np.clip(Ava@w, 0, None)
        errs.append(np.abs(p - y[va]))
    return float(np.concatenate(errs).mean())

print("prev18 (no hh):          ", round(loo_hh(SP, use_hh=False),3))
print("prev18 + hh dummies:     ", round(loo_hh(SP, use_hh=True, lam_h=100.0),3))
print("prev18 + hh (lam_h=30):  ", round(loo_hh(SP, use_hh=True, lam_h=30.0),3))
print("prev18 + hh (lam_h=300): ", round(loo_hh(SP, use_hh=True, lam_h=300.0),3))
print("prev18 + zw-interactions:", round(loo_hh(SP, use_hh=False, inter_zw=True),3))
print("prev18 + hh + zw-inter:  ", round(loo_hh(SP, use_hh=True, inter_zw=True),3))