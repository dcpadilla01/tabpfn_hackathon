import numpy as np, pandas as pd
v = agent_api.snapshot()
hh = v.households
print('households', hh if hh is None else type(hh))
# check via build_features what view.households is
def fn(view, day):
    print('day', day, 'week', view.week, 'n households', len(view.households))
    print(type(view.households))
    raise SystemExit
try:
    agent_api.build_features(fn)
except SystemExit:
    pass
except Exception as e:
    print('ERR', type(e).__name__, e)
