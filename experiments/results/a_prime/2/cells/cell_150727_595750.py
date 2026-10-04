import agent_api as A
v = A.snapshot(95)
print([a for a in dir(v) if not a.startswith('_')])
print(A.describe_tables())