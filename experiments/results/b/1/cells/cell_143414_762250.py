import pandas as pd, agent_api as A
base = A.load_saved('e001_history.parquet')
X = A.build_features(lambda v, d: v.transactions.groupby(v.households).size())  # placeholder; not used
