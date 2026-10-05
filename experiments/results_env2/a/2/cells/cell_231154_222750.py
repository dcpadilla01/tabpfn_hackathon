import pandas as pd, numpy as np
from agent_api import load_saved, save_table, KEYS
p4 = load_saved('pred_e004.parquet'); p5 = load_saved('pred_e005.parquet')
m = p4.merge(p5, on=KEYS, suffixes=('_e4','_e5'))
print(m.shape)
m['prediction'] = 0.5*m.prediction_e4 + 0.5*m.prediction_e5
out = m[KEYS + ['prediction']]
print(out['prediction'].describe().round(1))
print(save_table(out, 'pred_e006.parquet'))
