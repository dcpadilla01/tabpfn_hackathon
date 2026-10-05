
import agent_api as api, numpy as np, pandas as pd

df = api.load_saved("e011_discounts.parquet")
tt = api.train_targets().sort_values(["household_key","snapshot_day"])

# per-household aggregates over train snapshots
g = tt.groupby("household_key").future_spend_4w
agg = pd.DataFrame({"hh_sum": g.sum(), "hh_sumsq": (g.apply(lambda s: (s*s).sum())),
                    "hh_n": g.count()}).reset_index()
df = df.merge(agg, on="household_key", how="left")
df = df.merge(tt.rename(columns={"future_spend_4w":"tgt"})[["household_key","snapshot_day","tgt"]],
              on=["household_key","snapshot_day"], how="left")

is_tr = df.tgt.notna()
n = df.hh_n
# LOO mean/std over the household's train targets (val rows: full 13-target mean)
loo_mean = (df.hh_sum - df.tgt) / (n - 1)
loo_var = ((df.hh_sumsq - df.tgt**2) - (df.hh_sum - df.tgt)**2/(n-1)) / (n-2)
df["hh_mean_loo"] = np.where(is_tr & (n > 1), loo_mean, df.hh_sum/n)
df["hh_std_loo"]  = np.where(is_tr & (n > 2), np.sqrt(loo_var.clip(lower=0)), np.nan)
# causal (strictly earlier snapshots only); val rows -> full mean
tt["hh_mean_causal"] = (tt.groupby("household_key").future_spend_4w.cumsum() - tt.future_spend_4w) / \
                       (tt.groupby("household_key").future_spend_4w.cumcount()).replace(0, np.nan)
tt["hh_mean_last3"] = tt.groupby("household_key").future_spend_4w.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
df = df.merge(tt[["household_key","snapshot_day","hh_mean_causal","hh_mean_last3"]],
              on=["household_key","snapshot_day"], how="left")
df["hh_mean_causal"] = df.hh_mean_causal.fillna(df.hh_mean_loo)
df["hh_mean_last3"]  = df.hh_mean_last3.fillna(df.hh_mean_loo)

newcols = ["hh_mean_loo","hh_std_loo","hh_n","hh_mean_causal","hh_mean_last3"]
df = df.drop(columns=["tgt","hh_sum","hh_sumsq"])
print(df[newcols].describe().round(1))
print("rows:", len(df), "cols:", df.shape[1])
path = api.save_table(df, "e012_hh_target_enc")
print(path)
