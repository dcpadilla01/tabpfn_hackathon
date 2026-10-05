
import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e005_decay_gapcv.parquet')
tt = agent_api.train_targets()
m = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = m['future_spend_4w']
rows=[]
for c in t.columns:
    if c in ('household_key','snapshot_day'): continue
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    if ok.sum()>100:
        rows.append((c, float(np.corrcoef(x[ok], y[ok])[0,1])))
cs = pd.DataFrame(rows, columns=['feat','corr']).set_index('feat')
cs['abs'] = cs['corr'].abs()
print(cs.sort_values('abs', ascending=False).round(3).to_string())
# log-target correlations too
yl = np.log1p(y)
rows=[]
for c in cs.index:
    x = pd.to_numeric(m[c], errors='coerce')
    ok = x.notna() & y.notna()
    rows.append((c, float(np.corrcoef(x[ok], yl[ok])[0,1])))
csl = pd.DataFrame(rows, columns=['feat','corr_log']).set_index('feat')
print(csl.reindex(cs.sort_values('abs',ascending=False).index).round(3).to_string())
