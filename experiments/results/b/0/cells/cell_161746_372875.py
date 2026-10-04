import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403]

def ridge_eval(df, cols, fit_days, eval_days, lam=3.0):
    d = df[df.snapshot_day.isin(fit_days+eval_days)]
    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
    mu, sd = X.mean(), X.std().replace(0,1)
    Xs = ((X-mu)/sd).clip(-10,10).fillna(0).values
    Xs = np.column_stack([np.ones(len(Xs)), Xs])
    y = d["future_spend_4w"].values
    tr = d.snapshot_day.isin(fit_days).values
    b = np.linalg.solve(Xs[tr].T@Xs[tr]+lam*np.eye(len(cols)+1), Xs[tr].T@y[tr])
    p = Xs@b
    m = ~tr
    return np.mean(np.abs(p[m]-y[m]))

rb = A.load_saved("rich_behavioral.parquet")   # E003's 53 features
rfm = A.load_saved("rfm28.parquet")            # E001's 3 features
rb_m = rb.merge(tt, on=["household_key","snapshot_day"])
rfm_m = rfm.merge(tt, on=["household_key","snapshot_day"])

print("E001 harness=69.595 | my ridge lam sweep on [spend28,trips28,recency]:")
for lam in [0.3,1,3,10,30]:
    print(f"  lam={lam:5}: {ridge_eval(rfm_m, ['spend28','trips28','recency'], tr_days, [431], lam):.3f}")

rbf = [c for c in rb.columns if c not in ("household_key","snapshot_day")]
print("\nE003 harness=61.525 | my ridge on 53 rich_behavioral feats:")
for lam in [0.3,1,3,10,30]:
    print(f"  lam={lam:5}: {ridge_eval(rb_m, rbf, tr_days, [431], lam):.3f}")

# also eval on 403 for stability
print("\non 403 (fit<=375): rfm3:", {lam: round(ridge_eval(rfm_m, ['spend28','trips28','recency'], tr_days[:-1], [403], lam),3) for lam in [1,3,10]})
print("rich53 403:", {lam: round(ridge_eval(rb_m, rbf, tr_days[:-1], [403], lam),3) for lam in [1,3,10]})
