
import agent_api as A
import pandas as pd, numpy as np

# Build household-level long-run profile features from snapshot(459) history,
# then attach to all snapshot rows by household_key.
import agent_api as A
v = A.snapshot()  # capped at day 459
tx = v.table("transactions")
print("tx shape", tx.shape, "max day", tx.day.max())

g = tx.groupby("household_key")
prof = pd.DataFrame({
    "hh_spend_mean": g.spend_all if False else g.apply(lambda d: d.sales_value.sum()),
})
