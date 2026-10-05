
import pandas as pd
g = agent_api.load_saved("e016_grid.parquet")
gcols = [c for c in g.columns if c not in set(agent_api.load_saved("e008_level_shape.parquet").columns)]
print("E016 grid-only cols:", len(gcols))
print(gcols)
