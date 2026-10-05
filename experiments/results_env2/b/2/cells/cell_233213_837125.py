t = load_saved('e011_table.parquet')
print('e011_table', t.shape)
print('has keys:', 'household_key' in t.columns, 'snapshot_day' in t.columns)
cols = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print('n_feat', len(cols))
import re
groups = {}
for c in cols:
    g = re.split(r'(?=[A-Z])|_', c)[0]
    groups.setdefault(g, []).append(c)
for g in sorted(groups):
    cs = groups[g]
    print(g, len(cs), cs[:6])
print()
print('snapshot_days:', snapshot_days())
