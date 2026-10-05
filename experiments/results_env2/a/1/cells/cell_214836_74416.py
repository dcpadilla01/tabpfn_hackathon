
v = agent_api.snapshot(as_of_day=459)
print(type(v.households))
print(v.households.head() if hasattr(v.households, "head") else v.households[:5])
tx = v.table("transactions")
print(tx.shape, tx.day.min(), tx.day.max())
