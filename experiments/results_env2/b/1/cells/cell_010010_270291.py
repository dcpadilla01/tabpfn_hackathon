
import pandas as pd
for name in ["e008_level_shape.parquet", "e016_grid.parquet", "e013_union.parquet"]:
    df = agent_api.load_saved(name)
    print("==", name, df.shape)
    print(list(df.columns))
    print()
