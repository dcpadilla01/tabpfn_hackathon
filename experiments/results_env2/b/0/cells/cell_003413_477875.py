import pandas as pd, numpy as np
df = agent_api.load_saved('e019_rhythm.parquet')
new = [c for c in df.columns if c not in ('household_key','snapshot_day')][-15:]
tr = df[df.snapshot_day <= 431]; va = df[df.snapshot_day >= 459]
print(f"{'col':22s} {'nan%':>6s} {'tr_mean':>9s} {'va_mean':>9s} {'tr_std':>8s} {'va_std':>8s}")
for c in new:
    print(f"{c:22s} {tr[c].isna().mean()*100:5.1f}% {tr[c].mean():9.3f} {va[c].mean():9.3f} {tr[c].std():8.3f} {va[c].std():8.3f}")
print('dup check:', df.duplicated(['household_key','snapshot_day']).sum(), 'rows', len(df))
