import numpy as np, pandas as pd, agent_api as A

df = A.load_saved("e005_marketing.parquet")
tt = A.train_targets()
m = tt.merge(df, on=["household_key","snapshot_day"], how="left")
y = m["future_spend_4w"].astype(float).values

tr_days = A.snapshot_days()["train"]; va_days = A.snapshot_days()["validation"]
is_val = m["snapshot_day"].isin(va_days).values
is_tr = m["snapshot_day"].isin(tr_days).values

# Oracle: household's own median of y across TRAIN snapshots only (no val leakage)
tr = m[is_tr]
med = tr.groupby("household_key")["future_spend_4w"].median()
va = m[is_val]
pred_or = va["household_key"].map(med).fillna(tr["future_spend_4w"].median()).values
print("oracle (household median from train snaps) val MAE:", round(float(np.abs(va["future_spend_4w"]-pred_or).mean()),2))

# Oracle 2: household median including val snapshots (upper bound of persistence)
med2 = m.groupby("household_key")["future_spend_4w"].median()
pred_or2 = va["household_key"].map(med2).fillna(m["future_spend_4w"].median()).values
print("oracle (all snaps) val MAE:", round(float(np.abs(va["future_spend_4w"]-pred_or2).mean()),2))

feat_cols = [c for c in df.columns if c not in ("household_key","snapshot_day")]
X = m[feat_cols].copy()
for c in X.columns:
    if not np.issubdtype(X[c].dtype, np.number):
        X[c] = X[c].astype("category").cat.codes
X = X.replace([np.inf,-np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0).values.astype(float)

def ridge_eval(X, y, is_tr, is_val, alpha=100.0, standardize=True, logy=False):
    mu, sd = X[is_tr].mean(0), X[is_tr].std(0)+1e-9
    Z = (X-mu)/sd if standardize else X
    yt = np.log1p(y) if logy else y
    A_ = Z[is_tr]; b = yt[is_tr]
    d = A_.shape[1]
    G = A_.T@A_ + alpha*np.eye(d); G[-1,-1] -= alpha  # don't penalize intercept-ish? simple
    w = np.linalg.solve(G, A_.T@b)
    p = Z[is_val]@w
    if logy: p = np.expm1(np.clip(p,0,8))
    return float(np.abs(y[is_val]-p).mean()), float(np.abs(y[is_tr]-Z[is_tr]@w).mean() if not logy else np.abs(y[is_tr]-np.expm1(Z[is_tr]@w)).mean())

for a in [10,100,1000]:
    mae, trm = ridge_eval(X, y, is_tr, is_val, alpha=a)
    print(f"ridge alpha={a}: val MAE {mae:.2f} (train {trm:.2f})")
mae, trm = ridge_eval(X, y, is_tr, is_val, alpha=100, logy=True)
print("ridge log-target alpha=100: val MAE", round(mae,2), "(train", round(trm,2), ")")
