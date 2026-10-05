
import agent_api as A
import pandas as pd, numpy as np

v = A.snapshot()
tx = v.table("transactions")
g = tx.groupby("household_key")
prof = pd.DataFrame({
    "hh_spend_sum": g.sales_value.sum(),
    "hh_spend_mean": g.sales_value.mean(),
    "hh_spend_std": g.sales_value.std(),
    "hh_baskets": g.basket_id.nunique(),
    "hh_days_span": g.day.max() - g.day.min() + 1,
    "hh_first_day": g.day.min(),
})
prof["hh_spend_rate"] = prof.hh_spend_sum / prof.hh_days_span
prof["hh_active_frac"] = prof.hh_baskets / prof.hh_days_span
prof = prof.reset_index()
print(prof.shape)
print(prof.describe().round(2).T)

# distribution of hh_spend_rate: is it stable over time? compare first-half vs second-half rate per hh
tx["half"] = (tx.day > 229).astype(int)
h1 = tx[tx.half==0].groupby("household_key").sales_value.sum()
h2 = tx[tx.half==1].groupby("household_key").sales_value.sum()
cmp = pd.concat([h1.rename("h1"), h2.rename("h2")], axis=1).dropna()
cmp["r1"] = cmp.h1/229; cmp["r2"] = cmp.h2/230
print("corr of rates:", cmp.r1.corr(cmp.r2).round(3))
print("mean r1,r2:", cmp.r1.mean().round(3), cmp.r2.mean().round(3))
print("r2/r1 quantiles:", (cmp.r2/cmp.r1).quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
