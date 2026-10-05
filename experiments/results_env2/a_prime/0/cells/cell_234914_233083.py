
v = snapshot(95)
print([a for a in dir(v) if not a.startswith("_")])
print("households:", type(v.households), v.households)
print("day:", v.day, "week:", v.week)
