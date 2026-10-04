import pandas as pd, numpy as np, agent_api

names = ['e009_analog.parquet','e009_demo.parquet','e009_demo_mkt.parquet','e009_spline2p.parquet','e008_fwd_calendar.parquet']
info = {}
for nm in names:
    try:
        df = agent_api.load_saved(nm)
        info[nm] = df
        print(nm, df.shape, 'idx:', df.index.name)
    except Exception as e:
        print(nm, 'ERR', repr(e)[:100])

e008 = info['e008_fwd_calendar.parquet']
e008cols = set(e008.columns)
for nm, df in info.items():
    if nm == 'e008_fwd_calendar.parquet': continue
    extra = set(df.columns) - e008cols
    print(nm, 'n_extra_vs_e008:', len(extra))

cand = [nm for nm, df in info.items() if df.shape[1] == 141]
print('E009 candidates (141 cols):', cand)
if cand:
    b = info[cand[0]]
    print('E009 cols:', list(b.columns))

v = agent_api.snapshot()
tx = v.transactions
print(tx[['sales_value','coupon_disc','coupon_match_disc','retail_disc']].describe().T)
print('coupon_redemptions rows:', len(v.coupon_redemptions))
tt = agent_api.train_targets()
print(tt['future_spend_4w'].describe())
print('zero share:', float((tt['future_spend_4w']==0).mean()))
d = v.demographics
for c in ['classification_1','classification_3','classification_4','homeowner_desc','kid_category_desc']:
    print(c, sorted(map(str, d[c].dropna().unique().tolist())))