
import agent_api as A

for name in ['e014_base','e014_gbm_oob','e014_stack','e013_denoise']:
    try:
        t = A.load_saved(name + '.parquet')
        print('===', name, t.shape)
        print(t.columns.tolist()[:40])
        print('snapshot_days:', sorted(t.snapshot_day.unique()))
        print(t.head(3))
        print()
    except Exception as e:
        print(name, 'ERR', e)
