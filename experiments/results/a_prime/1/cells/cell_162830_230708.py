import pandas as pd, numpy as np
import agent_api as A

base = A.load_saved('e016_smoothed.parquet')
print('e016_smoothed (=E014 base):', base.shape)
feat = [c for c in base.columns if c not in ('household_key','snapshot_day')]
print('n feat:', len(feat))
keys = ['since','recen','trip','zero','window','nw','dept','brand','price','tenur','gap','iv_','overdue','max','cv','std','store','dow','weekend']
print([c for c in feat if any(k in c.lower() for k in keys)])

for nm in ['e017_disc_seasonal.parquet','e017_v2.parquet']:
    try:
        t = A.load_saved(nm)
        extra = [c for c in t.columns if c not in set(base.columns)]
        print(nm, t.shape, 'extra cols:', extra[:40])
    except Exception as e:
        print(nm, 'ERR', repr(e))

def probe(view, sd):
    print('sd', sd, 'n_hh', len(view.households), 'tx_rows', len(view.transactions),
          'has_products', hasattr(view,'products'), 'hh_type', type(view.households))
    return pd.DataFrame(index=pd.Index(list(view.households), name='household_key')).assign(x=1.0)
df = A.build_features(probe)
print('probe table:', df.shape)
print(A.snapshot_days())
