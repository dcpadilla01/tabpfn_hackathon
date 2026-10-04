import agent_api, pandas as pd, numpy as np

demo = agent_api.snapshot().demographics
cats = [c for c in demo.columns if c!='household_key']
demo = demo[['household_key']+cats].copy()
for c in cats:
    demo[c] = demo[c].astype(str).fillna('NA')

mkt = agent_api.load_saved('mkt_v2.parquet')

def fn(view, snapshot_day):
    base = mkt[mkt.snapshot_day==snapshot_day].set_index('household_key')
    D = base.join(demo.set_index('household_key'), how='left')
    D['d_has_demo'] = D['classification_1'].notna().astype(int)
    for c in cats:
        D[c] = D[c].fillna('NA')
    return D.drop(columns=['snapshot_day'], errors='ignore')

out = agent_api.build_features(fn)
out = out.reset_index()
print(out.shape, flush=True)
print(out.columns.tolist()[:10], "...", flush=True)
path = agent_api.save_table(out, 'e009_demo.parquet')
print(path, flush=True)
