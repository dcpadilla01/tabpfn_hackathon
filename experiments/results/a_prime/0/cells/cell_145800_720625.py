import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
season = load_saved('season_v1.parquet')
season = season.drop(columns=['index'], errors='ignore').rename(columns={'tenure':'yr_tenure'})
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns])
tt = train_targets()
m = tt.merge(e3, on=['household_key','snapshot_day'], how='left').merge(season, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
ok = m.p13.notna().values
print("rows with p13:", ok.sum())
print("corr(y,p13):", round(float(np.corrcoef(m.p13[ok], y[ok])[0,1]),3),
      " corr(y,spend_84):", round(float(np.corrcoef(m.spend_84[ok], y[ok])[0,1]),3))
print("corr(y,log1p p13):", round(float(np.corrcoef(np.log1p(m.p13[ok]), y[ok])[0,1]),3))
print("corr(y,log1p spend84):", round(float(np.corrcoef(np.log1p(m.spend_84[ok]), y[ok])[0,1]),3))
print("corr(y,p13_ratio):", round(float(np.corrcoef(m.p13_ratio[ok], y[ok])[0,1]),3))
print("corr(y,p13_div_s28):", round(float(np.corrcoef(m.p13_div_s28[ok], y[ok])[0,1]),3))
print("mean y | p13==0:", round(y[ok & (m.p13.values<=1)],1) if False else round(float(y[ok & (m.p13.values<=1)].mean()),1),
      " mean y | p13>1:", round(float(y[ok & (m.p13.values>1)].mean()),1))

def prep(df, med=None):
    X = df.copy()
    if med is None: med = X.median()
    for c in X.columns:
        col = X[c]
        if col.dtype.kind not in 'ifbu':
            col = pd.Series(pd.factorize(col)[0], index=X.index).astype(float)
        X[c] = col.astype(float).replace([np.inf,-np.inf], np.nan).fillna(med[c])
    return X.values, med
bcols = [c for c in e3.columns if c not in ('household_key','snapshot_day')]
scols = ['p13','p13_avg2','p13_ratio','p13_div_s28','p13_zero','yr_tenure']
Xb,_ = prep(m[bcols]); Xs,_ = prep(m[scols])
Xa = np.hstack([Xb, Xs])
def ridge_mae(X, tr, va, lam=100.0):
    mu, sd = X[tr].mean(0), X[tr].std(0)+1e-9
    Z = (X-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
    A = Zt[tr].T@Zt[tr] + lam*np.eye(Zt.shape[1]); A[-1,-1]-=lam
    w = np.linalg.solve(A, Zt[tr].T@y[tr])
    return float(np.mean(np.abs(y[va]-Zt[va]@w)))
for vd in [431, 403]:
    tr = np.where(m.snapshot_day.isin([x for x in [95,123,151,179,207,235,263,291,319,347,375,403,431] if x!=vd]).values)[0]
    va = np.where((m.snapshot_day==vd).values)[0]
    print(f"val={vd}: base {ridge_mae(Xb,tr,va):.2f}  +season {ridge_mae(Xa,tr,va):.2f}")
tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values)[0]
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + 100*np.eye(Zt.shape[1]); A[-1,-1]-=100
w = np.linalg.solve(A, Zt[tr].T@y[tr])
res = y - Zt@w
print("partial corr p13|base:", round(float(np.corrcoef(m.p13.fillna(m.p13.median()), res)[0,1]),3))
print("partial corr log1p p13|base:", round(float(np.corrcoef(np.log1p(m.p13.fillna(0)), res)[0,1]),3))