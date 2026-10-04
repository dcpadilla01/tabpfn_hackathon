import agent_api as A, numpy as np, pandas as pd
v = A.snapshot(431)
print(type(v.households), repr(v.households)[:200])
try:
    print(len(v.households))
except Exception as e:
    print("len err", e)
print(type(v.day), repr(v.day)[:100])
print(type(v.week), repr(v.week)[:100])
