import agent_api
v = agent_api.snapshot(95)
attrs = [a for a in dir(v) if not a.startswith('_')]
print(attrs)
print('day:', v.day, 'week:', v.week)
print(agent_api.KEYS, agent_api.TARGET)
