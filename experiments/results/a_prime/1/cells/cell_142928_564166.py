import agent_api as api
v = api.snapshot(459)
print([a for a in dir(v) if not a.startswith('_')])
print(type(v.transactions))
print(api.KEYS, api.TARGET)
print([a for a in dir(api) if not a.startswith('_')])
