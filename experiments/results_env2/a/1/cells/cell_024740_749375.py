import agent_api as api
import pandas as pd, numpy as np
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
for name in ['e005_preds','e011_preds','e004_preds','e007_preds','e010_preds','e009_preds','e012_preds','repro_e5','e005_newfeats','lagfeats2']:
    try:
        df = api.load_saved(name+'.parquet')
        print(name, df.shape, list(df.columns)[:8])
        if 'prediction' in df.columns:
            print('   days:', sorted(df.snapshot_day.unique())[:20], 'pred mean %.2f median %.2f' % (df.prediction.mean(), df.prediction.median()))
    except Exception as e:
        print(name, 'ERR', type(e).__name__, e)
