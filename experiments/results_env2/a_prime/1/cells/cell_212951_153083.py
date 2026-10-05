
v = agent_api.snapshot(95)
print([a for a in dir(v) if not a.startswith('_')])
print("KEYS:", agent_api.KEYS, "TARGET:", agent_api.TARGET)
print(agent_api.snapshot_days())
print(agent_api.describe_tables())
