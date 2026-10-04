import agent_api as A, pandas as pd
v = A.snapshot()
print('dir view:', [x for x in dir(v) if not x.startswith('_')])
print('KEYS:', A.KEYS)
print('TARGET:', A.TARGET)
print(A.describe_tables())
