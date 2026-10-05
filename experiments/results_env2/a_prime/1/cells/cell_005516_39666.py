import agent_api as api
import pandas as pd, numpy as np

e15 = api.load_saved('e015_stack.parquet')
print(list(e15.columns))
