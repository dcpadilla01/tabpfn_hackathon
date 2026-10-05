import agent_api as A, pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

te = A.load_saved("e007_te.parquet")
tt = A.train_targets()
df = tt.merge(te, on=["household_key","snapshot_day"], how="left")

# per-snapshot missing te
print(df.groupby("snapshot_day")["te_hh_shrunk"].apply(lambda s: s.isna().sum()))

# single-feature baselines with fallback chain
y = df[A.TARGET].values
def mae(p): return float(np.mean(np.abs(y-p)))
med = np.median(y)
p_te = df["te_hh_shrunk"].fillna(df["spend_4w_lag1"]).fillna(med).values
print("te chain MAE:", round(mae(p_te),2))
p_lag1 = df["spend_4w_lag1"].fillna(med).values
print("lag1 MAE:", round(mae(p_lag1),2))
for w in np.linspace(0,1,11):
    print(f"w={w:.1f}", round(mae(w*p_te+(1-w)*p_lag1),2))

# zero-spend rows: what predicts them?
z = y==0
print("\nzero rows:", z.sum(), "share", round(z.mean(),3))
for c in ["days_since_last","active_4w","trend_4_28","spend_4w","nbask_4w","spend_4w_lag1","gap_mean_112","tenure_days"]:
    print(f"{c:18s} zero-mean={df.loc[z,c].mean():8.2f}  nonzero-mean={df.loc[~z,c].mean():8.2f}")
