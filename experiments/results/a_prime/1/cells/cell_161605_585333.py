import agent_api as A
import pandas as pd

e12 = A.load_saved("e012_dorm.parquet")
e13 = A.load_saved("e013_peers.parquet")
e14 = A.load_saved("e014_demo_l13fix.parquet")

c12 = set(e12.columns); c13 = set(e13.columns); c14 = set(e14.columns)
print("e12 - e14:", sorted(c12 - c14))
print("e14 - e12:", sorted(c14 - c12))
print("e13 - e14:", sorted(c13 - c14))
print("e14 - e13:", sorted(c14 - c13))
print(e12.shape, e13.shape, e14.shape)