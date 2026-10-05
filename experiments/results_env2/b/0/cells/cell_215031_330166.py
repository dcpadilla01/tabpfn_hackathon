import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table("transactions")
    hh = view.households
    tx = tx[tx.household_key.isin(hh)]
    # lag1: spend in [s-28, s-1]  (what the target at snapshot s-28 would have been)
    def win(lo, hi):
        w = tx[(tx.day >= lo) & (tx.day <= hi)]
        return w.groupby("household_key").sales_value.sum()
    lag1 = win(s-28, s-1)
    lag2 = win(s-56, s-29)
    lag3 = win(s-84, s-57)
    lag4 = win(s-112, s-85)
    ly   = win(s-28-364, s-1-364)   # same 4w window last year
    df = pd.DataFrame({"lag1": lag1, "lag2": lag2, "lag3": lag3, "lag4": lag4, "ly_lag": ly})
    df = df.reindex(hh).fillna(0.0)
    return df

feats = agent_api.build_features(fn)
tt = agent_api.train_targets()
m = tt.merge(feats, on=["household_key","snapshot_day"])
tr = m[m.snapshot_day <= 431]
for c in ["lag1","lag2","lag3","lag4","ly_lag"]:
    print(c, "corr:", round(np.corrcoef(tr[c], tr.future_spend_4w)[0,1],3),
          "mean:", round(tr[c].mean(),1))
print("target mean/std:", round(tr.future_spend_4w.mean(),1), round(tr.future_spend_4w.std(),1))
# how often is lag1 exactly the target?
print("lag1==target frac:", (tr.lag1==tr.future_spend_4w).mean())
# zero structure
print("target zero frac:", (tr.future_spend_4w==0).mean(), "lag1 zero frac:", (tr.lag1==0).mean())
