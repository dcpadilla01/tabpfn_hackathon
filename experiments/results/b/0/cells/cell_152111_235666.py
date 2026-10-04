v = snapshot()
print('day:', v.day, 'week:', v.week)
print('n hh:', len(v.households))
t = v.transactions
print('tx rows:', len(t), 'day max:', t.day.max())