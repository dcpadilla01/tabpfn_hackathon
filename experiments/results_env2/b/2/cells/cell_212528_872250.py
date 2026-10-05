import agent_api, pandas as pd, numpy as np
t = agent_api.train_targets()
print(t.shape)
print(t.future_spend_4w.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print("zero share:", (t.future_spend_4w==0).mean())
print(t.head())