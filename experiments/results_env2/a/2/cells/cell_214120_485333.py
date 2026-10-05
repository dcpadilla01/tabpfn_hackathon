v = agent_api.snapshot(431)
print(type(v))
print("households:", v.households)
print("day:", v.day, "week:", v.week)
tx = v.table("transactions")
print(tx.day.max(), tx.shape[0])
# derive households: first purchase >= 84 days earlier
firsts = tx.groupby("household_key")["day"].min()
hh = firsts[firsts <= 431-84].index
print(len(hh))
