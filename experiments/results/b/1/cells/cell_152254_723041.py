
import pandas as pd, numpy as np
import agent_api as A

tt = A.train_targets()
e8 = A.load_saved("e008_decomp2.parquet")
e1 = A.load_saved("e001_history.parquet")
m = tt.merge(e8, on=["household_key","snapshot_day"]).merge(
    e1[["household_key","snapshot_day","spend_7","spend_56","spend_84","spend_182","spend_365",
        "nb_28","nb_56","nb_all","avg_basket_28","spend_prev28","trend28","qty_28","nprod_28",
        "spend_ly28","weekly_rate_84","snapshot_day_index","week_of_year"]],
    on=["household_key","snapshot_day"])

# 1) drift of target and key features across snapshot days
print("target mean by snapshot_day:")
print(m.groupby("snapshot_day")["future_spend_4w"].agg(["mean","median","count"]).round(1))
print("\nspend_28 mean by snapshot (drift check):")
print(m.groupby("snapshot_day")["spend_28"].mean().round(1))
print("\np13 mean by snapshot:")
print(m.groupby("snapshot_day")["p13"].mean().round(3))

# 2) ridge fit on E008 features -> residual correlation with candidate new features
feats = [c for c in e8.columns if c not in ("household_key","snapshot_day")]
X = m[feats].copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median())
y = m["future_spend_4w"].values
Xm = X.values
Xm = (Xm - Xm.mean(0)) / (Xm.std(0) + 1e-9)
# simple ridge via normal equations
lam = 1.0
G = Xm.T @ Xm + lam*np.eye(Xm.shape[1])
b = np.linalg.solve(G, Xm.T @ y)
pred = Xm @ b
res = y - pred
print("\nridge-train MAE:", np.abs(res).mean().round(2))

cand = ["spend_7","spend_56","spend_84","spend_182","spend_365","nb_28","nb_56","nb_all",
        "avg_basket_28","spend_prev28","trend28","qty_28","nprod_28","spend_ly28","weekly_rate_84",
        "snapshot_day_index","week_of_year"]
for c in cand:
    v = pd.to_numeric(m[c], errors="coerce").fillna(0).values
    v = (v - v.mean())/(v.std()+1e-9)
    print(f"corr(resid, {c:18s}) = {np.corrcoef(res, v)[0,1]: .4f}")

# 3) zero-rate features
print("\nzero frac by snapshot:")
print(m.assign(z=(m.future_spend_4w==0)).groupby("snapshot_day")["z"].mean().round(3))
