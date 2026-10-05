
import pandas as pd
for name in ["e008_level_shape.parquet", "e016_grid.parquet", "e013_union.parquet"]:
    df = agent_api.load_saved(name)
    print("==", name, df.shape)
    print(list(df.columns))
    print()


# ---- cell ----

import pandas as pd
g = agent_api.load_saved("e016_grid.parquet")
gcols = [c for c in g.columns if c not in set(agent_api.load_saved("e008_level_shape.parquet").columns)]
print("E016 grid-only cols:", len(gcols))
print(gcols)


# ---- cell ----

import pandas as pd

e013 = agent_api.load_saved("e013_union.parquet")
e016 = agent_api.load_saved("e016_grid.parquet")

grid_keep = ['g_w0r','g_w1r','g_w2r','g_wmean','g_wmed','g_wstd','g_wmin','g_wmax',
             'g_wzero','g_wslope','g_ew6','g_ew85','g_shr1','g_shr3','g_dlog_01',
             'g_dlog_rm','g_lt0','g_lt1','g_lt2','g_lt3','g_lt4','g_lt5','g_tmean',
             'g_tzero','g_dsl','g_expact','g_dlog_seas']

sub = e016[['household_key','snapshot_day'] + grid_keep]
m = e013.merge(sub, on=['household_key','snapshot_day'], how='inner', validate='one_to_one')
print("e013:", e013.shape, "merged:", m.shape)
assert len(m) == len(e013) == 36426
print("n feature cols:", m.shape[1] - 2)
path = agent_api.save_table(m, "e019_grand_gridshape.parquet")
print(path)
