
import agent_api as A
for n in ['e017_v2.parquet','e017_disc_seasonal.parquet','e016_smoothed.parquet']:
    t = A.load_saved(n)
    print('='*15, n, t.shape)
    cols=list(t.columns)
    for i in range(0,len(cols),8): print(' | '.join(cols[i:i+8]))
