import agent_api as A, pandas as pd, numpy as np
tt = A.train_targets()
f = A.load_saved("weekly_history.parquet")
m = tt.merge(f, on=['household_key','snapshot_day'], how='left')
wc = [c for c in f.columns if c.startswith(('ws_','wt_','wl_','wblk_','wyoy','wmean26','wmax26','wstd26','wnz26','wshare_last4','wslope26','wks_since_active'))]
cor = m[wc].corrwith(m['future_spend_4w'])
cor = cor.sort_values(key=np.abs, ascending=False)
print(cor.head(30))