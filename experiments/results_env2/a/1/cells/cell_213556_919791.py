import agent_api as A
print(A.snapshot_days())
print(A.KEYS, A.TARGET)
s = A.snapshot()
print("day", s.day, "week", s.week)
print("n households", len(s.households))
print(A.describe_tables())
