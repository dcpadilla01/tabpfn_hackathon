import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"]).sales_value.sum().reset_index()
    def wsum(lo, hi):
        w = g[(g.day >= lo) & (g.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["c_fwd_1_28"] = wsum(s+1, s+28)
    out["c_bwd_27_0"] = wsum(s-27, s)
    return out.fillna(0.0)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
for c in ["c_fwd_1_28","c_bwd_27_0"]:
    print(c, "NaNs:", m[c].isna().sum(), "unique:", m[c].nunique())
d = (m.c_fwd_1_28 - m.future_spend_4w).abs()
print("fwd_1_28: |diff|<0.01 frac:", (d<0.01).mean(), " max diff:", d.max())
print(m[["future_spend_4w","c_fwd_1_28","c_bwd_27_0"]].head(8).round(2))
