import numpy as np, pandas as pd

def seasonal_fn(view, snapshot_day):
    d = snapshot_day
    keys = view.households.index if isinstance(view.households, pd.DataFrame) else pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.household_key.isin(keys)]
    out = pd.DataFrame(index=keys)
    # tenure per household (first purchase day)
    first = tx.groupby('household_key').day.min()
    tenure = d - first
    # year-ago same-window: days d-363 .. d-336 (future window shifted -364)
    ya = tx[(tx.day > d-364) & (tx.day <= d-336)]
    p13 = ya.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0)
    ok13 = (tenure >= 364)  # full window observed
    p13 = p13.where(ok13)
    out['p13'] = p13
    # year-ago window one block earlier (stability): d-391..d-364
    ya2 = tx[(tx.day > d-392) & (tx.day <= d-364)]
    p14 = ya2.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0).where(tenure >= 392)
    out['p13_avg2'] = ((p13.fillna(0)+p14.fillna(0))/2).where(ok13)
    # trailing year-ago 4 weeks: d-391..d-364 is p14; trailing relative to ya window is d-391..d-364
    # ratio: year-ago next-window vs year-ago trailing-window (p14)
    out['p13_ratio'] = p13 / p14.replace(0, np.nan)
    # ratio to current level
    t28 = tx[tx.day > d-28]
    s28 = t28.groupby('household_key').sales_value.sum().reindex(keys).fillna(0.0)
    out['p13_div_s28'] = p13 / s28.replace(0, np.nan)
    out['p13_zero'] = (p13.fillna(0) <= 1.0).astype(float).where(ok13)
    out['tenure'] = tenure
    return out

feats = build_features(seasonal_fn)
print("built:", feats.shape)
print("p13 non-NaN share by snapshot_day:")
print(feats.groupby('snapshot_day').p13.apply(lambda s: s.notna().mean()).round(2).to_string())
save_table(feats.reset_index(), 'season_v1')

# proxy test: E003 base + seasonal block, ridge, temporal val=431 and val=403
base = load_saved('e005_full_plus_mix.parquet')
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns])
print("E003 feats:", e3.shape[1]-2)
tt = train_targets()
m = tt.merge(e3, on=['household_key','snapshot_day'], how='left').merge(feats.reset_index(), on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)
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
scols = ['p13','p13_avg2','p13_ratio','p13_div_s28','p13_zero','tenure']
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
# corr of y with p13 among rows with p13
ok = m.p13.notna().values
print("rows with p13:", ok.sum(), "corr(y,p13):", round(float(np.corrcoef(m.p13[ok], y[ok])[0,1]),3),
      "corr(y,spend_84):", round(float(np.corrcoef(m.spend_84[ok], y[ok])[0,1]),3))
# partial given base
tr = np.where(m.snapshot_day.isin([95,123,151,179,207,235,263,291,319,347,375,403,431]).values)[0]
mu, sd = Xb[tr].mean(0), Xb[tr].std(0)+1e-9
Z = (Xb-mu)/sd; Zt = np.hstack([Z, np.ones((len(Z),1))])
A = Zt[tr].T@Zt[tr] + 100*np.eye(Zt.shape[1]); A[-1,-1]-=100
w = np.linalg.solve(A, Zt[tr].T@y[tr])
res = y - Zt@w
print("partial corr p13|base:", round(float(np.corrcoef(m.p13.fillna(m.p13.median()), res)[0,1]),3))