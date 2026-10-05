import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")

E7 = agent_api.load_saved("e007_te.parquet")
JUNK = ["day","week","tenure","snap_day","snap_week","index","snapshot_day"]

def make_feats(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(list(view.households), name="household_key")
    e7s = E7[E7["snapshot_day"] == s].set_index("household_key")
    e7s = e7s.drop(columns=[c for c in JUNK if c in e7s.columns])
    e7s = e7s.reindex(hh)
    tt = agent_api.train_targets()
    tt = tt[tt["snapshot_day"] < s].sort_values(["household_key", "snapshot_day"])
    gm = tt["future_spend_4w"].mean() if len(tt) else np.nan
    g = tt.groupby("household_key")["future_spend_4w"]
    lagcols = ["lag%d" % k for k in range(1, 14)]
    S = pd.DataFrame({c: g.shift(int(c[3:])) for c in lagcols})
    oh = pd.DataFrame(index=tt.index)
    for hl, name in [(1.5, "oh_ewm15"), (4.0, "oh_ewm4")]:
        w = 0.5 ** ((np.arange(1, 14) - 1.0) / hl)
        M = S.notna().values.astype(float) * w[None, :]
        den = M.sum(1, keepdims=True); den[den == 0] = 1.0
        oh[name] = (S.fillna(0).values * (M / den)).sum(1)
    oh["oh_n"] = g.cumcount().astype(float)
    Z = S[["lag1", "lag2", "lag3", "lag4", "lag5", "lag6"]]
    oh["oh_zero_share6"] = (Z == 0).sum(1) / Z.notna().sum(1).clip(lower=1)
    oh["oh_zero_any6"] = ((Z == 0).sum(1) > 0).astype(float)
    oh["oh_min3"] = S[["lag1", "lag2", "lag3"]].min(1)
    oh["oh_max3"] = S[["lag1", "lag2", "lag3"]].max(1)
    oh["oh_trend3"] = S["lag1"] - S["lag3"]
    m3 = S[["lag1", "lag2", "lag3"]]
    oh["oh_cv3"] = m3.std(1) / m3.mean(1).replace(0, np.nan)
    oh["oh_ewm15_x_active"] = oh["oh_ewm15"] * (1 - oh["oh_zero_share6"].fillna(0))
    oh["oh_ewm15_x_zeroflag"] = oh["oh_ewm15"] * (1 - oh["oh_zero_any6"])
    oh = oh.reindex(hh)
    nohist = oh["oh_n"].fillna(0) == 0
    oh.loc[nohist, ["oh_zero_share6","oh_zero_any6","oh_min3","oh_max3","oh_trend3","oh_cv3",
                    "oh_ewm15_x_active","oh_ewm15_x_zeroflag"]] = np.nan
    for c in ["oh_ewm15", "oh_ewm4", "oh_min3", "oh_max3"]:
        oh[c] = oh[c].fillna(gm)
    df = e7s.join(oh)
    demo = view.table("demographics")
    d2 = demo.set_index("household_key").astype(str)
    df = df.join(d2, how="left")
    df["has_demo"] = df.index.isin(set(demo["household_key"])).astype(float)
    df["snapshot_day"] = float(s)
    return df

res = agent_api.build_features(make_feats)
print("shape:", res.shape)
print("rows/snapshot:", res.groupby("snapshot_day").size().to_dict())
print("n feature cols:", res.shape[1] - 2)
print("oh_ewm15 nonnull:", round(res.oh_ewm15.notna().mean(), 3), " te_hh_mean nonnull:", round(res.te_hh_mean.notna().mean(), 3))
print("any NaN keys:", res.household_key.isna().any(), res.snapshot_day.isna().any())
print(res[["oh_ewm15","oh_n","te_hh_mean","snapshot_day"]].describe().round(2))
path = agent_api.save_table(res, "e011_outcomehist.parquet")
print("saved:", path)
