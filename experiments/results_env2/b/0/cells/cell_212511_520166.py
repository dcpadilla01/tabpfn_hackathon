import agent_api as A
v = A.snapshot(459)
print(type(v.households), v.day, v.week)
print([a for a in dir(v) if not a.startswith('_')])
# try history
h = A.history(2375, 459)
print(h.shape)
print(h.head())
