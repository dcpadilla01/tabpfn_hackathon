import agent_api as api
e = api.load_saved('e018_basestab.parquet')
print('e018', e.shape)
cols = list(e.columns)
print('demo-like cols in e018:', [c for c in cols if any(k in c.lower() for k in ('class','home','kid','demo','age','size'))])
print('first cols:', cols[:20])
print('last cols:', cols[-12:])
sn = api.load_saved('season.parquet'); print('season', sn.shape, list(sn.columns))
md = api.load_saved('mkt_demo.parquet'); print('mkt_demo', md.shape, list(md.columns))
v = api.snapshot()
print('view has demographics attr:', hasattr(v, 'demographics'))
if hasattr(v, 'demographics'):
    d = v.demographics
    print('demo', d.shape, list(d.columns))
    print(d.head(3).to_string())
print('KEYS', api.KEYS, 'TARGET', api.TARGET)
print('households type', type(v.households))
try:
    print(v.households.head())
except Exception as ex:
    print('hh head err', ex, v.households[:5])