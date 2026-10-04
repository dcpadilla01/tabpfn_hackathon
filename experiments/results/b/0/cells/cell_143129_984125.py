import agent_api, numpy as np, pandas as pd
snap = agent_api.snapshot()
print(type(snap))
print([a for a in dir(snap) if not a.startswith('_')])
print("day:", snap.day, "week:", snap.week)
hh = snap.households
print("households:", type(hh), hh[:10])
