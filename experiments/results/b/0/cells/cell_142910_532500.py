import agent_api as api
v = api.snapshot(95)
print([a for a in dir(v) if not a.startswith('_')])
print(api.describe_tables())
