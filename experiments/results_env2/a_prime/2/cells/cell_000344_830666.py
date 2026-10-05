
import numpy as np, pandas as pd

# --- facts: campaign timing coverage ---
c = snapshot(459).campaigns
print('campaigns n=', len(c), 'min start', c.start_day.min(), 'max end', c.end_day.max())
print(c.sort_values('start_day').to_string())
for d in [95, 207, 319, 347, 431]:
    cc = snapshot(d).campaigns
    print(d, 'n campaigns', len(cc), 'starts', sorted(cc.start_day.tolist())[:6])
r = snapshot(459).coupon_redemptions
print('redemptions day range', r.day.min(), r.day.max(), 'n', len(r))

tt = train_targets()
print(tt.future_spend_4w.describe().round(2))
print('zero share', round((tt.future_spend_4w==0).mean(),3), 'n rows', len(tt))

# --- ridge proxy on E011 base ---
base = load_saved('e011_discounts.parquet')
df = base.merge(tt, on=['household_key','snapshot_day'])
tr_days = snapshot_days()['train']; va_days = snapshot_days()['validation']
feats = [c for c in base.columns if c not in ('household_key','snapshot_day')]

def prep(d):
    X = d[feats].copy()
    for c in feats:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype('category').cat.codes.astype(float)
            X[c] = X[c].where(X[c] >= 0, np.nan)
    return X.astype(float)

tr_df, va_df = df[df.snapshot_day.isin(tr_days)], df[df.snapshot_day.isin(va_days)]
Xtr_raw, Xva_raw = prep(tr_df), prep(va_df)
mu, sg = Xtr_raw.mean(), Xtr_raw.std().replace(0,1)+1e-9
Xtr = ((Xtr_raw-mu)/sg).fillna(0.0).values; Xva = ((Xva_raw-mu)/sg).fillna(0.0).values
ytr, yva = tr_df.future_spend_4w.values, va_df.future_spend_4w.values

def ridge_eval(lam):
    A = np.hstack([Xtr, np.ones((len(Xtr),1))]); Av = np.hstack([Xva, np.ones((len(Xva),1))])
    D = np.eye(A.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A.T@A + lam*D, A.T@ytr)
    return np.abs(A@w-ytr).mean(), np.abs(Av@w-yva).mean()

for lam in [1,3,10,30,100,300,1000,3000]:
    trm, vam = ridge_eval(lam)
    print('lam', lam, 'train MAE', round(trm,3), 'val MAE', round(vam,3))
