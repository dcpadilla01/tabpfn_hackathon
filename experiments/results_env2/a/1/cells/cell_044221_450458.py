import pandas as pd, numpy as np

for f in ['oof_e5','repro_e5','e016_allpreds','e016_sqpreds','e016_held','e004_new','e004_preds','e005_preds','e011_preds','e018_preds']:
    try:
        d = load_saved(f + '.parquet')
        print('===', f, d.shape)
        print(' cols:', list(d.columns))
        if 'snapshot_day' in d.columns:
            print(' days:', sorted(d.snapshot_day.unique()))
    except Exception as e:
        print(f, 'ERR', type(e).__name__, str(e)[:80])

# train targets coverage
tt = train_targets()
print('\ntarget days:', sorted(tt.snapshot_day.unique()))
print('rows per day:'); print(tt.groupby('snapshot_day').size())
