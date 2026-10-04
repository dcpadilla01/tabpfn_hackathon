import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
tt = train_targets()
m = tt.merge(base, on=['household_key','snapshot_day'], how='left')
y = m.future_spend_4w.values.astype(float)

def mae(p): return float(np.mean(np.abs(y-p)))
print("median pred:", mae(np.full(len(y), np.median(y))))
print("mean pred:", mae(np.full(len(y), y.mean())))
for c in ['spend_84','spend_28','spend_56','x_exp4w','spend_364']:
    x = m[c].fillna(m[c].median()).values.astype(float)
    print(c, "identity MAE:", round(mae(x),2), end='  ')
    A = np.vstack([x, np.ones(len(x))]).T
    coef,*_ = np.linalg.lstsq(A, y, rcond=None)
    print("linfit MAE:", round(mae(A@coef),2))

# per-snapshot-day mean target as predictor (calendar only)
daymean = m.groupby('snapshot_day').future_spend_4w.mean()
print("day-mean pred MAE:", round(mae(daymean.reindex(m.snapshot_day).values),2))

# distribution of y vs spend_84 ratio
r = y / np.maximum(m.spend_84.values.astype(float),1)
print("y/spend_84 ratio quantiles:", np.nanpercentile(r,[10,25,50,75,90]).round(2))
print("corr y with log1p(spend_84):", np.corrcoef(np.log1p(m.spend_84.values), y)[0,1])