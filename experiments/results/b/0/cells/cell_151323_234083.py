import agent_api
v = agent_api.snapshot(459)
print('households:', v.households)
print('day:', v.day, 'week:', v.week)
print('tx:', type(v.transactions))
print(v.transactions.head(2))
print('products:', type(v.products))
print(v.products.head(2))
print([a for a in dir(v) if not a.startswith('_')])
