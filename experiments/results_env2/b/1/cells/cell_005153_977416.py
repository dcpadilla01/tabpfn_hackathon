import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

t13 = agent_api.load_saved('e013_union.parquet')   # E008+E006+E007 union, best table (60.703)
print('E013', t13.shape)

df = t13.copy()
# --- new feature block: non-linear/interaction level features ---
eps = 1e-9
# 1) sqrt (concave) transforms of top heavy-tailed level features -> closer to conditional median
for c in ['sp28','sp84','sp364','g_wmean','z_med4w_hist','sp28_rate','sp364_rate']:
    if c in df.columns:
        df['sq_'+c] = np.sqrt(np.clip(df[c].astype(float), 0, None))
# 2) empirical-Bayes shrunk expected next-4w spend: rate shrunk toward global daily rate
m_day = df['sp364'].astype(float).sum() / (df['sp364'].notna().sum()*364.0)
for w, k in [(84,56),(168,56),(364,112)]:
    sp = df['sp%d'%w].astype(float)
    df['exp_eb%d'%w] = 28.0 * (sp.fillna(0) + k*m_day) / (w + k)
# 3) zero-probability-adjusted expectation
zr = df['z_zero_rate_hist'].astype(float).fillna(0.5).clip(0,1)
df['expz84'] = df['exp_eb84'] * (1-zr)
df['expz364'] = df['exp_eb364'] * (1-zr)
# 4) recency-weighted and cadence-weighted level interactions
dsl = df['days_since_last'].astype(float).fillna(999)
df['i_rec_level']  = df['sp28'].astype(float).fillna(0) * np.exp(-dsl/28.0)
df['i_trips_lvl']  = df['sp28_rate'].astype(float).fillna(0) * df['trips28'].astype(float).fillna(0)
df['i_act_lvl']    = df['sp84_rate'].astype(float).fillna(0) * df['nact84'].astype(float).fillna(0)
df['i_med_gw']     = df['z_med4w_hist'].astype(float).fillna(0) * df['g_wmean'].astype(float).fillna(0)
df['i_ew_gw']      = df['g_ew6'].astype(float).fillna(0) * df['g_wmean'].astype(float).fillna(0)
# 5) blend of the two best level estimators
df['lvl_blend'] = 0.5*df['exp_eb84'].fillna(0) + 0.25*df['exp_eb364'].fillna(0) + 0.25*df['g_wmean'].fillna(0)

newc = [c for c in df.columns if c not in t13.columns]
print('new features:', newc, len(newc))
X = df[newc].astype(float).values
print('nonfinite in new block:', (~np.isfinite(X)).sum())
df[newc] = np.where(np.isfinite(X), X, 0.0)
print('final shape', df.shape)
path = agent_api.save_table(df, 'e018_nonlin.parquet')
print(path)
