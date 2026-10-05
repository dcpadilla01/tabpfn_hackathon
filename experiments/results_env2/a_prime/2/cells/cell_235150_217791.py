import agent_api, pandas as pd, numpy as np
v = agent_api.snapshot(459)
print("view type:", type(v))
print([x for x in dir(v) if not x.startswith("_")])
print("day:", v.day, "week:", v.week)
hh = v.households
print("households type:", type(hh))
try:
    import itertools
    print("first few:", list(itertools.islice(hh, 5)))
except Exception as e:
    print("iter err", repr(e))
tr = v.transactions
print("tr type:", type(tr), getattr(tr, "shape", None))
