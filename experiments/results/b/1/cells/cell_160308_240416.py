import agent_api as A, pandas as pd, numpy as np
def dbg(view, s):
    t = view.transactions
    print("snap", s, "cols", list(t.columns)[:15], "shape", t.shape)
    print("r cols", list(view.coupon_redemptions.columns))
    print("c cols", list(view.campaigns.columns))
    print("ct cols", list(view.campaign_targets.columns))
    fd = t.groupby("household_key", sort=False).day.min()
    print("fd ok", len(fd))
    return pd.DataFrame({"x": np.zeros(len(fd))}, index=fd.index)
out = A.build_features(dbg)
print(out.shape)
