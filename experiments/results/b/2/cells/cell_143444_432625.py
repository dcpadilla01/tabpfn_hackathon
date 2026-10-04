from agent_api import snapshot
v = snapshot()
print(type(v.households))
print(v.households[:5] if hasattr(v.households,'__getitem__') else v.households.head())
