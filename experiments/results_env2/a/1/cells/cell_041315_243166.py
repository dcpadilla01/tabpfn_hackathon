import agent_api as api, pandas as pd, numpy as np
tt = api.train_targets()
print("Per-snapshot-day intrinsic difficulty (MAE of predicting the day's median):")
for d, g in tt.groupby('snapshot_day'):
    med = g['future_spend_4w'].median()
    print(f"day {d}: n={len(g)} mean={g['future_spend_4w'].mean():6.1f} std={g['future_spend_4w'].std():6.1f} medianMAE={np.abs(g['future_spend_4w']-med).mean():6.1f} zero_share={(g['future_spend_4w']==0).mean():.3f}")
