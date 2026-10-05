import agent_api, numpy as np, pandas as pd

tt = agent_api.train_targets()

def prep(path):
    t = agent_api.load_saved(path)
    feat = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    m = t.merge(tt, on=['household_key','snapshot_day'], how='left')
    isval = m.future_spend_4w.isna().values
    Xdf = m[feat].copy()
    for c in feat:
        if str(Xdf[c].dtype) in ('object','category','bool'):
            Xdf[c] = Xdf[c].astype('category').cat.codes.replace(-1, np.nan).astype(float)
        else:
            Xdf[c] = Xdf[c].astype(float)
    return feat, Xdf.values, isval

print('=== E009 TE features sanity ===')
f9, X9, v9 = prep('e009_target_enc.parquet')
te_cols = [c for c in f9 if c.startswith('te_') or 'te' in c.lower()][:12]
print('te-like cols:', te_cols)
idx = [f9.index(c) for c in te_cols]
for c,i in zip(te_cols, idx):
    a, b = X9[~v9, i], X9[v9, i]
    print(f'{c:28s} nanT={np.isnan(a).mean():.3f} nanV={np.isnan(b).mean():.3f} meanT={np.nanmean(a):8.2f} meanV={np.nanmean(b):8.2f} stdT={np.nanstd(a):7.2f} stdV={np.nanstd(b):7.2f}')

print()
print('=== E008 train vs validation-row distribution shifts (top 15) ===')
f8, X8, v8 = prep('e008_level_shape.parquet')
shifts = []
for j,c in enumerate(f8):
    a, b = X8[~v8, j], X8[v8, j]
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a)<10 or len(b)<10: continue
    sd = np.sqrt(np.nanvar(X8[:,j]) + 1e-9)
    shifts.append((abs(a.mean()-b.mean())/sd, c, a.mean(), b.mean()))
shifts.sort(reverse=True)
for s,c,ma,mb in shifts[:15]:
    print(f'{c:24s} shift={s:.2f}  train={ma:9.3f} val={mb:9.3f}')
print()
print('=== E004 marketing features: nan/means train vs val (first 12) ===')
f4, X4, v4 = prep('e004_marketing.parquet')
mk = [c for c in f4 if c not in f8][:12]
for c in mk:
    i = f4.index(c)
    a, b = X4[~v4, i], X4[v4, i]
    print(f'{c:24s} nanT={np.isnan(a).mean():.3f} nanV={np.isnan(b).mean():.3f} meanT={np.nanmean(a):7.3f} meanV={np.nanmean(b):7.3f}')
