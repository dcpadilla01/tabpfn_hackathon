import pandas as pd, numpy as np
t1 = agent_api.load_saved('recency_agg.parquet')
t2 = agent_api.load_saved('dept_mix_recency.parquet')
print('E001 cols:', list(t1.columns))
print()
extra = [c for c in t2.columns if c not in t1.columns]
print('E002 extra cols (first 30):', extra[:30], '... total', len(extra))
tt = agent_api.train_targets()
print()
print(tt.future_spend_4w.describe())
print('share zero target:', (tt.future_spend_4w==0).mean())

def probe(view, d):
    base = agent_api.load_saved('dept_mix_recency.parquet')
    base = base[base.snapshot_day==d].set_index('household_key')
    return base.iloc[:, :2]
df = agent_api.build_features(probe)
print('probe ok:', df.shape)


# ---- cell ----
import numpy as np, pandas as pd

def fn(view, d):
    hh = pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.household_key.isin(hh)]
    w = int(view.week)
    out = pd.DataFrame(index=hh)

    # weekly spend / trips over past complete weeks (w-16 .. w-1)
    g = tx.groupby(['household_key','week_no']).agg(sp=('sales_value','sum'), tr=('basket_id','nunique'))
    spw = g['sp'].unstack(fill_value=0.0)
    trw = g['tr'].unstack(fill_value=0.0)
    weeks = [x for x in (w-i for i in range(1,17)) if x >= 1]
    spw = spw.reindex(index=hh, columns=weeks).fillna(0.0)
    trw = trw.reindex(index=hh, columns=weeks).fillna(0.0)
    S = spw.to_numpy(); T = trw.to_numpy(); nc = S.shape[1]

    for k in range(1, min(8, nc)+1):
        out[f'ts_sw_{k}'] = S[:, nc-k]
    out['ts_sum4'] = S[:, -4:].sum(1)
    out['ts_sum8'] = S[:, -8:].sum(1)
    out['ts_max8'] = S[:, -8:].max(1)
    out['ts_std8'] = S[:, -8:].std(1)
    out['ts_zero8'] = (S[:, -8:] == 0).sum(1)
    out['ts_share_top8'] = out['ts_max8']/(out['ts_sum8']+1.0)
    x = np.arange(nc-8, nc, dtype=float); x = x - x.mean()
    out['ts_slope8'] = S[:, -8:] @ x / (x @ x)
    out['ts_ratio_4_8'] = out['ts_sum4']/(out['ts_sum8']-out['ts_sum4']+1.0)
    out['ts_sum16'] = S.sum(1)*(16.0/nc)
    out['ts_zero16_share'] = (S == 0).sum(1)/float(nc)
    r = np.arange(nc)[::-1]  # recency of each column
    out['ts_ewma2'] = (S @ (0.5**(r/2.0)))/(0.5**(r/2.0)).sum()
    out['ts_ewma4'] = (S @ (0.5**(r/4.0)))/(0.5**(r/4.0)).sum()
    out['ts_ratio_w'] = out['ts_sum4']/(S[:, -8:-4].sum(1)+1.0)
    out['ts_trips4'] = T[:, -4:].sum(1)
    out['ts_trips8'] = T[:, -8:].sum(1)

    # 28-day periodic lags (aligned 4-week windows)
    for name, lo, hi in [('p1', d-27, d), ('p2', d-55, d-28), ('p3', d-83, d-56), ('p4', d-111, d-84)]:
        m = (tx.day >= lo) & (tx.day <= hi)
        out[f'ts_{name}'] = tx[m].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    mean234 = (out['ts_p2']+out['ts_p3']+out['ts_p4'])/3.0
    out['ts_p1_diff'] = out['ts_p1'] - mean234
    out['ts_p1_ratio'] = out['ts_p1']/(mean234+5.0)

    # trip-gap regularity over last 84 days
    bd = tx.groupby(['household_key','basket_id'])['day'].min().reset_index()
    bd = bd[bd.day >= d-83].sort_values(['household_key','day'])
    bd['gap'] = bd.groupby('household_key')['day'].diff()
    gg = bd.groupby('household_key')['gap']
    out['ts_gap_mean'] = gg.mean().reindex(hh)
    out['ts_gap_std'] = gg.std().reindex(hh)
    out['ts_gap_max'] = gg.max().reindex(hh)
    out['ts_gap_cv'] = out['ts_gap_std']/out['ts_gap_mean']
    return out

new = agent_api.build_features(fn)
print('new feats:', new.shape, 'nan cols:', new.isna().sum().gt(0).sum())
base = agent_api.load_saved('dept_mix_recency.parquet')
merged = base.merge(new.reset_index(), on=['household_key','snapshot_day'], how='inner')
print('merged:', merged.shape)
assert merged.groupby('snapshot_day').size().reindex(agent_api.snapshot_days()['train']+agent_api.snapshot_days()['validation']).notna().all()
path = agent_api.save_table(merged, 'temporal_structure.parquet')
print(path)

# ---- cell ----
import pandas as pd
t = agent_api.load_saved('temporal_structure.parquet')
if 'index' in t.columns: t = t.drop(columns=['index'])
print(t.shape)
print(agent_api.save_table(t, 'temporal_structure.parquet'))