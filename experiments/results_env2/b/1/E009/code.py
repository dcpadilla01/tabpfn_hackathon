import pandas as pd, numpy as np, agent_api

base = agent_api.load_saved("e008_level_shape.parquet")
tt = agent_api.train_targets()
print("base", base.shape, "targets", tt.shape)

# Household-level target encoding from TRAIN targets only.
# For any validation snapshot (>=459) these are strictly past outcomes,
# i.e. the household's own historical average 4-week spend + its trend.
tt = tt.sort_values(["household_key", "snapshot_day"])
g = tt.groupby("household_key")["future_spend_4w"]
enc = pd.DataFrame({
    "te_mean": g.mean(),
    "te_median": g.median(),
    "te_std": g.std(),
    "te_min": g.min(),
    "te_max": g.max(),
    "te_count": g.count(),
})
last3 = tt.groupby("household_key").tail(3).groupby("household_key")["future_spend_4w"].mean().rename("te_mean_last3")
enc = enc.join(last3)

def slope(sub):
    x = sub["snapshot_day"].to_numpy(float); y = sub["future_spend_4w"].to_numpy(float)
    if len(x) < 2 or x.var() == 0: return np.nan
    return np.cov(x, y)[0,1] / x.var()
enc["te_slope"] = tt.groupby("household_key").apply(slope)

enc = enc.reset_index()
df = base.merge(enc, on="household_key", how="left")
print("merged", df.shape, "nulls:", df[["te_mean","te_mean_last3","te_slope"]].isna().sum().to_dict())

# sanity: association on train rows
m = df.merge(tt, on=["household_key","snapshot_day"])
print("corr te_mean vs target:", np.corrcoef(m["te_mean"], m["future_spend_4w"])[0,1].round(3))
print("corr te_mean_last3 vs target:", np.corrcoef(m["te_mean_last3"], m["future_spend_4w"])[0,1].round(3))
print(m[["future_spend_4w","te_mean","te_mean_last3"]].describe().round(1))

path = agent_api.save_table(df, "e009_target_enc.parquet")
print(path)
