import agent_api as api
v = api.snapshot(as_of_day=431)
print(type(v.households), repr(v.households)[:200])
try:
    hh = v.households()
    print('callable ->', type(hh), len(hh), hh[:5])
except Exception as e:
    print('not callable:', e)
print('view.day:', v.day, 'view.week:', v.week)
tx = v.table('transactions')
print('tx:', tx.shape)
