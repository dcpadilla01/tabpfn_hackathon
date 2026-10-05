import agent_api as A
import numpy as np
v = A.snapshot(459)
h = v.households
print(type(h), getattr(h, 'shape', None))
print(repr(h)[:300])
