import agent_api as api
v = api.snapshot(459)
print(type(v.households))
print(v.households if hasattr(v.households,'shape') else v.households[:5])
print(v.day, v.week)
h = api.history(1)
print(h.head(3), h.columns.tolist())
