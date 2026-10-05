import agent_api as A
print([a for a in dir(A) if not a.startswith('_')])
print(A.KEYS, A.TARGET)
print(A.describe_tables())
