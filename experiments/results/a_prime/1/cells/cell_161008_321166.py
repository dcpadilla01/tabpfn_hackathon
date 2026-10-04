import agent_api as A, pandas as pd, numpy as np

# Build E014: E010 + E011's demographic columns (independent mutations of E003 combined)
e10 = A.load_saved("e010_l13fix.parquet")
e11 = A.load_saved("e011_demo.parquet")
demo_cols = ["classification_1","classification_2","classification_3","classification_4",
             "classification_5","homeowner_desc","kid_category_desc","has_demographics"]
out = e10.merge(e11[["household_key","snapshot_day"]+demo_cols], on=["household_key","snapshot_day"], how="left")
out["has_demographics"] = out["has_demographics"].fillna(0).astype(int)
print(out.shape, "| demo cols:", [c for c in out.columns if c in demo_cols])
p = A.save_table(out, "e014_demo_l13fix.parquet")
print(p)
