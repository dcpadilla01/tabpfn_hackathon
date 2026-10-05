import agent_api as A, pandas as pd, numpy as np
te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")
y = df[A.TARGET].values

# Baselines: what do simple predictors achieve in MAE?
def mae(p): return np.mean(np.abs(y - p))
print("global median:", round(mae(np.median(y)),2))
print("global mean:", round(mae(y.mean()),2))
for c in ["te_hh_shrunk","te_hh_mean","spend_4w","spend_8w","spend_4w_lag1","te_prior"]:
    if c in df: print(f"{c:14s}", round(mae(df[c].values),2))

# blend of te and spend_4w (manual search) as upper-bound check
best=None
for w in np.linspace(0,1,21):
    p = w*df["te_hh_shrunk"].values + (1-w)*df["spend_4w"].values
    m = mae(p)
    if best is None or m<best[0]: best=(m,w)
print("best blend te/spend4w:", round(best[0],2), "w=",round(best[1],2))

# error decomposition of te_hh_shrunk: where is loss?
p = df["te_hh_shrunk"].values
e = np.abs(y-p)
df["err"] = e
print("\nMAE by target bucket:")
df["yb"] = pd.qcut(y, 10, duplicates="drop")
print(df.groupby("yb", observed=True).agg(n=("err","size"), mae=("err","mean"), pred=("te_hh_shrunk","mean"), true=(A.TARGET,"mean")).round(1))
