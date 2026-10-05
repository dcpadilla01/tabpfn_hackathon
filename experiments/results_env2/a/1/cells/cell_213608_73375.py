import agent_api as A
s = A.snapshot()
attrs = [a for a in dir(s) if not a.startswith('_')]
print(attrs)
print("day", s.day, "week", s.week)
tr = s.table("transactions")
print(tr.shape)
print(tr['day'].max())
