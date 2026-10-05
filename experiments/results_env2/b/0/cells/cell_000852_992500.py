import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved("e013_stationary.parquet")
print("base:", base.shape)

def fn(view, sd):
    tx = view.table("transactions")
    idx = view.households.index
    lg = np.log1p
    def wsum(w):
        return tx[tx.day > sd - w].groupby("household_key").sales_value.sum()
    s7, s14, s28, s56, s84 = (wsum(w) for w in (7, 14, 28, 56, 84))
    age = (sd - tx.day).astype(float)
    def dec(hl):
        return pd.Series(tx.sales_value.values * np.power(0.5, age.values / hl),
                         index=tx.index).groupby(tx.household_key).sum()
    d28, d112 = dec(28), dec(112)
    R = lambda s: s.reindex(idx).fillna(0.0)
    s7, s14, s28, s56, s84, d28, d112 = map(R, (s7, s14, s28, s56, s84, d28, d112))
    f = pd.DataFrame(index=idx)
    f["spike_7"]  = lg(s7)  - lg(s84 * 7 / 84)
    f["spike_14"] = lg(s14) - lg(s84 * 14 / 84)
    f["spike_28"] = lg(s28) - lg(s84)
    f["spike_56"] = lg(s56) - lg(s84 * 56 / 84)
    f["spike_dec"] = lg(d28) - lg(d112 * 28 / 112)
    f["spike_28_pos"] = f["spike_28"].clip(lower=0)
    f["spike_28_neg"] = f["spike_28"].clip(upper=0)
    rev = d28 * np.exp(-0.7 * f["spike_28_pos"])
    f["rev_rate_28"] = np.where(d28 > 0, rev, 0.0)
    return f

out = agent_api.build_features(fn)
print("new feats:", out.shape, list(out.columns))
merged = base.merge(out, on=["household_key", "snapshot_day"], how="left")
print("merged:", merged.shape, "NaN frac:", round(merged.isna().mean().mean(), 4))
print(merged[["spike_28","spike_dec","rev_rate_28"]].describe().round(3))
path = agent_api.save_table(merged, "e017_reversion.parquet")
print("saved:", path)
