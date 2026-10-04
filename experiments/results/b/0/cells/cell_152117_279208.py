v = snapshot()
print('day:', v.day, 'week:', v.week)
t = v.transactions
print(type(t))
print('tx rows:', len(t), 'day max:', t.day.max())
print('n hh:', v.households.nunique())