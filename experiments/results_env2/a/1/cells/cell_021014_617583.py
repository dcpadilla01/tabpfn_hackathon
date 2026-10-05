import pandas as pd, numpy as np
from agent_api import load_saved
F = load_saved("allF.parquet")
print([c for c in F.columns])
