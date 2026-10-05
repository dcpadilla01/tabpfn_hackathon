import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"], as_index=False).sales_value.sum()
    def wsum(lo,hi):
        w = g[(g.day>=lo)&(g.day<=hi)]
        return w.groupby("household_key").sales_value.sum()
    # aligned 28-day blocks: lag1 = [s-27, s] == target at snapshot s-28
    df = pd.DataFrame({
        "lag1": wsum(s-27, s),
        "lag2": wsum(s-55, s-28),
        "lag3": wsum(s-83, s-56),
        "lag4": wsum(s-111, s-84),
        "ly_lag": wsum(s-391, s-364),
    }).reindex(hh).fillna(0.0)
    df["lag_trend"] = df.lag1 - df.lag2
    df["lag_ratio"] = df.lag1/(df.lag2+1.0)
    df["lag_mean4"] = df[["lag1","lag2","lag3","lag4"]].mean(axis=1)
    df["lag_std4"] = df[["lag1","lag2","lag3","lag4"]].std(axis=1).fillna(0.0)
    df["ly_ratio"] = df.lag1/(df.ly_lag+1.0)
    return df

feats = agent_api.build_features(fn)
print("feats:", feats.shape)
base = agent_api.load_saved("e006_zero_inflation.parquet")
print("base:", base.shape)
m = base.merge(feats, on=["household_key","snapshot_day"], how="left")
print("merged:", m.shape, "dup rows:", int(m.duplicated(["household_key","snapshot_day"]).sum()),
      "NaN lag1:", int(m.lag1.isna().sum()))
# quick sanity on last train snapshot
tt = agent_api.train_targets()
chk = tt[tt.snapshot_day==431].merge(m[["household_key","snapshot_day","lag1","lag2"]],
                                     on=["household_key","snapshot_day"])
print("corr(lag1,target@431):", round(float(np.corrcoef(chk.lag1, chk.future_spend_4w)[0,1]),3))
print("corr(lag2,target@431):", round(float(np.corrcoef(chk.lag2, chk.future_spend_4w)[0,1]),3))
path = agent_api.save_table(m, "e007_ar_lags.parquet")
print(path)
