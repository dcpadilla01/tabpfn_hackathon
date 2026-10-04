import agent_api, numpy as np, pandas as pd

t = agent_api.load_saved('e009_ewma_longlags.parquet')
keys = ['household_key','snapshot_day']
fc = [c for c in t.columns if c not in keys + ['index']]
print('col names containing tlag/ewma/spend/trip/day/week/active:')
import re
pat = re.compile(r'tlag|ewma|spend|trip|line|day|week|active|life|yoy|recen|gap|zero|wblk|ws_|wl_|wmean|sw_|rdecay|ts_')
print(sorted([c for c in fc if pat.search(c)])[:120])
print('\nALL cols (%d):' % len(fc))
print(fc)