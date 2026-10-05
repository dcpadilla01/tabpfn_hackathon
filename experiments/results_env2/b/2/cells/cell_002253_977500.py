import pandas as pd, numpy as np
v = agent_api.snapshot(95)
print("households:", repr(v.households)[:200])
tx = v.table("transactions")
print("tx hh dtype:", tx.household_key.dtype, repr(tx.household_key.iloc[0]))
print("tx day dtype:", tx.day.dtype)
