import pandas as pd, numpy as np
v = agent_api.snapshot(95)
hh = v.households
print(type(hh), len(hh), hh[:3])
tx = v.table("transactions")
print(tx.household_key.dtype, tx.household_key.iloc[0], type(tx.household_key.iloc[0]))
cr = v.table("coupon_redemptions")
print(cr.household_key.dtype, cr.household_key.iloc[0] if len(cr) else None)
