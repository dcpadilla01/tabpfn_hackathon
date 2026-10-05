import pandas as pd, numpy as np
from agent_api import load_saved, train_targets
t = load_saved("e016_merged.parquet")
print("E016 table:", t.shape)
cols = list(t.columns)
print("n feature cols:", len(cols)-2)
print(cols[:50])
print(cols[-40:])
tt = train_targets()
print(tt["future_spend_4w"].describe())
print("zero share:", (tt["future_spend_4w"]==0).mean())
