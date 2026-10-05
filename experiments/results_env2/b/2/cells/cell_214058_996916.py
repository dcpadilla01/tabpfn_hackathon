v = agent_api.snapshot(459)
print(type(v.households))
try:
    print(len(v.households()))
except Exception as e:
    print("err", e)
print([a for a in dir(v) if not a.startswith('_')])
print(agent_api.KEYS, agent_api.TARGET)