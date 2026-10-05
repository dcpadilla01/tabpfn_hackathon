
import numpy as np, pandas as pd

rfm = agent_api.load_saved('rfm_v1.parquet')
print('rfm', rfm.shape, rfm.columns.tolist())

def fn(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(view.households, name='household_key')
    tx = view.table('transactions')
    out = pd.DataFrame(index=hh)
    # spend per aligned 28-day window, last 14 windows (w13 = same window last year)
    for k in range(1, 15):
        lo, hi = s - 28*k, s - 28*(k-1)
        w = tx[(tx['day'] > lo) & (tx['day'] <= hi)]
        out[f'spend_w{k}'] = w.groupby('household_key')['sales_value'].sum().reindex(hh, fill_value=0.0)
        if k <= 6:
            out[f'trips_w{k}'] = w.groupby('household_key')['basket_id'].nunique().reindex(hh, fill_value=0)
    W = [f'spend_w{k}' for k in range(1, 7)]
    eps = 1.0
    out['trend_w1_w2'] = out['spend_w1'] / (out['spend_w2'] + eps)
    out['trend_w12_w34'] = (out['spend_w1'] + out['spend_w2']) / (out['spend_w3'] + out['spend_w4'] + 2*eps)
    out['mean_w1_w6'] = out[W].mean(axis=1)
    out['std_w1_w6'] = out[W].std(axis=1)
    out['cv_w1_w6'] = out['std_w1_w6'] / (out['mean_w1_w6'] + eps)
    out['spend_w13_ratio'] = out['spend_w13'] / (out['mean_w1_w6'] + eps)
    g = tx.groupby('household_key')
    tot = g['sales_value'].sum().reindex(hh, fill_value=0.0)
    first = g['day'].min().reindex(hh)
    tenure = (s - first).clip(lower=1)
    out['spend_per_28d_all'] = tot / (tenure / 28.0)
    out['tenure_28d'] = tenure / 28.0
    return out

traj = agent_api.build_features(fn)
print('traj', traj.shape)

merged = rfm.merge(traj, on=['household_key', 'snapshot_day'], how='inner')
print('merged', merged.shape)
assert len(merged) == len(rfm) == len(traj)

tt = agent_api.train_targets()
m = merged.merge(tt, on=['household_key', 'snapshot_day'])
num = [c for c in merged.columns if c not in ('household_key', 'snapshot_day')]
cor = m[num + ['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w')
cor = cor.reindex(cor.abs().sort_values(ascending=False).index)
print(cor.head(25))

path = agent_api.save_table(merged, 'rfm_traj_v1')
print(path)
