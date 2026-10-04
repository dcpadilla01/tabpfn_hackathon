
import agent_api as A
t = A.load_saved('e018_union_full.parquet')
cols = list(t.columns)
for i in range(0, len(cols), 8):
    print(' | '.join(cols[i:i+8]))
