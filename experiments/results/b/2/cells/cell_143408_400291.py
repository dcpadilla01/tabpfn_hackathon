from agent_api import snapshot
v = snapshot()
print([a for a in dir(v) if not a.startswith('_')])
print(v.products.columns.tolist() if hasattr(v,'products') else 'no products')
print(v.transactions.shape)
