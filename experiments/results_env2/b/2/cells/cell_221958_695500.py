import agent_api as A
v = A.snapshot(459)
print(type(v))
print([a for a in dir(v) if not a.startswith("_")])
print("households:", type(v.households), v.households if v.households is None else str(v.households)[:200])
print("day:", v.day, "week:", v.week)
tx = v.table("transactions")
print("tx shape", tx.shape, "max day", tx.day.max())
print(A.describe_tables())
