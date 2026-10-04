import agent_api, pandas as pd, numpy as np
t = agent_api.load_saved('e009_ewma_longlags.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w.values.astype(float)
d = df.snapshot_day.astype(int).values
va = d==431; tr = d<=403

hh_tr = set(df.loc[tr,'household_key']); hh_va = set(df.loc[va,'household_key'])
print('households: train rows %d, val rows %d, overlap %d, val-only %d'%(len(hh_tr),len(hh_va),len(hh_tr&hh_va),len(hh_va-hh_tr)))

tr_df = df.loc[tr]
hh_med = tr_df.groupby('household_key')['future_spend_4w'].median()
hh_cnt = tr_df.groupby('household_key')['future_spend_4w'].size()
va_df = df.loc[va].copy()
med_map = va_df.household_key.map(hh_med)
cnt_map = va_df.household_key.map(hh_cnt).fillna(0)
med_glob = np.median(y[tr])
va_df['hh_med'] = med_map.fillna(med_glob)
va_df['hh_shr'] = (med_map*cnt_map + med_glob*3.0)/(cnt_map+3.0)
print('global median MAE %.2f'%np.abs(med_glob-y[va]).mean())
print('per-household median MAE %.2f'%np.abs(va_df.hh_med-y[va]).mean())
print('shrunk(k=3) MAE %.2f'%np.abs(va_df.hh_shr-y[va]).mean())
tl = df.loc[va,'tlag_mean'].values
for w in [0.3,0.5,0.7]:
    print('blend tlag+hh_med w=%.1f MAE %.2f'%(w, np.abs(w*tl+(1-w)*va_df.hh_med.values-y[va]).mean()))

print(df[['household_key','snapshot_day','index']].head(8).to_string())
print(df.groupby('snapshot_day')['index'].agg(['min','max']).head(4).to_string())
