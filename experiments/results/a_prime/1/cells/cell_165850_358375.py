
import agent_api as A
t = A.load_saved('e019_everything.parquet')
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))
