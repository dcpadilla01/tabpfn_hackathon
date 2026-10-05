import agent_api as api
v = api.snapshot(459)
print(type(v.households), v.households if v.households is None else len(v.households))
print([a for a in dir(v) if not a.startswith("_")])
print("day", v.day, "week", v.week)
