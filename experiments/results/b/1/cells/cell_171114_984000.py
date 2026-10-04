import agent_api as api
import pandas as pd, numpy as np

for t in ['e015_base','e013_union','e009_macro','cand_new','e006_seq_gaps']:
    df = api.load_saved(t + '.parquet')
    print('==', t, df.shape)
    print(list(df.columns))
    print()

bf = api.baseline_features()
print('== baseline_features', bf.shape)
print(list(bf.columns))
print(bf.head(3))
print()

for t in ['e001_history','e002_marketing','e003_dept_mix','e004_long_hist','e005_seasonal_peer','e007_new','e008_decomp2','e010_composite','e011_display','e014_base']:
    df = api.load_saved(t + '.parquet')
    print('==', t, df.shape, list(df.columns))
    print()