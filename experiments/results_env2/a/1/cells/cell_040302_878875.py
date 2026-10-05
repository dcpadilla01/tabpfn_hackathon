import numpy as np, pandas as pd
oof = load_saved("oof_e5.parquet")
held = load_saved("e016_held.parquet")
print("oof cols:", list(oof.columns))
print("held cols:", list(held.columns))
print("allF cols tail:", [c for c in load_saved("allF.parquet").columns][-8:])
