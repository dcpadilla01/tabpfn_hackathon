import agent_api as A
import pandas as pd, numpy as np

tt = A.train_targets()
harness = {"e015_base":60.761,"e013_union":61.337,"e009_macro":61.647,"e011_display":61.680,
           "e012_full":61.471,"e008_decomp2":61.711,"e007_new":62.794,"e001_history":63.025,
           "e014_base":63.242,"e010_composite":63.939,"e004_long_hist":63.982,"e006_seq_gaps":64.050,
           "e003_dept_mix":64.175,"e002_marketing":64.676,"e005_seasonal_peer":67.852}
for nm in harness:
    df = A.load_saved(nm + ".parquet")
    dupcols = pd.Index(df.columns)[pd.Index(df.columns).duplicated()].tolist()
    M = df.merge(tt, on=["household_key","snapshot_day"], how="inner")
    Mdup = pd.Index(M.columns)[pd.Index(M.columns).duplicated()].tolist()
    if dupcols or Mdup:
        print(nm, "dup in df:", dupcols, "| dup in M:", Mdup)
    # check index duplication
    print(nm, "idx dup:", M.index.duplicated().sum(), "shape", M.shape)