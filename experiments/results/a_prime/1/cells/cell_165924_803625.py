
import agent_api as A
ev = A.load_saved('e019_everything.parquet')   # 155 = E018 minus 31 timing
fm = A.load_saved('e019_full_merged.parquet')  # 137 pruned merge
th = A.load_saved('e018_timing_hazard.parquet')
print('timing_hazard:', th.shape, list(th.columns))
extra = [c for c in fm.columns if c not in ev.columns]
dropped = [c for c in ev.columns if c not in fm.columns]
print('\nIN pruned-merge but NOT in e019_everything (%d):' % len(extra))
print(extra)
print('\nIN e019_everything but NOT in pruned-merge (%d):' % len(dropped))
print(dropped)
print('\nunion size check:', len(set(ev.columns)|set(th.columns)), 'ev+th overlap:', len(set(ev.columns)&set(th.columns)))
