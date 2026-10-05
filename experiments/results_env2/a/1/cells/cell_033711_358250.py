import pandas as pd, numpy as np
r = agent_api.load_saved('repro_e5.parquet')
print(r.shape, list(r.columns))
print(r.head())
p5 = agent_api.load_saved('e005_preds.parquet'); p11 = agent_api.load_saved('e011_preds.parquet')
m = p5.merge(p11, on=['household_key','snapshot_day'], suffixes=('_e5','_e11'))
print('corr e5 vs e11 preds: %.4f' % m.prediction_e5.corr(m.prediction_e11))
print('mean diff', (m.prediction_e5-m.prediction_e11).abs().mean())
if 'p13' in r.columns and 'p11' in r.columns:
    print('repro p13 vs p11 corr %.4f' % r.p13.corr(r.p11))
# blend check (no scoring, just sanity of spread)
for w in [0,0.25,0.5,0.75,1.0]:
    b = w*m.prediction_e5 + (1-w)*m.prediction_e11
    print('w_e5=%.2f blend mean %.1f std %.1f' % (w, b.mean(), b.std()))