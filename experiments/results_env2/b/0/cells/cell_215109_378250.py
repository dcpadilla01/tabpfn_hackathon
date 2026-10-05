import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    g = tx.groupby(["household_key","day"]).sales_value.sum().reset_index()
    # daily spend series per household, then window sums for candidate definitions
    def wsum(lo, hi):
        w = g[(g.day >= lo) & (g.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    out = pd.DataFrame(index=hh)
    out["c_fwd_1_28"] = wsum(s+1, s+28)      # stated definition
    out["c_fwd_0_27"] = wsum(s, s+27)
    out["c_fwd_0_28"] = wsum(s, s+28)
    out["c_bwd_27_0"] = wsum(s-27, s)
    out["c_fwd_1_29"] = wsum(s+1, s+29)
    return out.fillna(0.0)

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
for c in ["c_fwd_1_28","c_fwd_0_27","c_fwd_0_28","c_bwd_27_0","c_fwd_1_29"]:
    eq = (m[c] == m.future_spend_4w).mean()
    print(c, "exact-match frac:", round(eq,4), " corr:", round(np.corrcoef(m[c], m.future_spend_4w)[0,1],4))
