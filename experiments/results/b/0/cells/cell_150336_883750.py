import warnings; warnings.filterwarnings('ignore')
v = snapshot()
tx = v.transactions
hh = v.households
print('n hh', type(hh), hh is None)
print('len hh', len(hh) if hh is not None else None)