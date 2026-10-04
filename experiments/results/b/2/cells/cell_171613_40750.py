
import agent_api as A
for name in ['e014_base','e014_gbm_oob','e014_stack']:
    t = A.load_saved(name + '.parquet')
    print('===', name, t.shape)
    print(t.columns.tolist()[-30:])
