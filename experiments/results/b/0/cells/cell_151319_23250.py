import agent_api
v = agent_api.snapshot(459)
print(type(v.households), len(v.households), v.households[:3])
print(type(v.day), v.day)
print(type(v.week), v.week)
print(type(v.transactions), v.transactions.shape)
print(type(v.products), v.products.shape)
