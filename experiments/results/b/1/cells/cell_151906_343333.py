import agent_api, pandas as pd, numpy as np

def feats(view, T):
    hh = view.households
    if isinstance(hh, pd.DataFrame):
        keys = pd.Index(hh['household_key'].unique(), name='household_key')
    else:
        keys = pd.Index(pd.Series(hh).unique(), name='household_key')
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(keys)) & (tx.day > T - 730)].copy()
    tx['k'] = (T - tx.day) // 28
    gw = tx.groupby(['household_key','k']).sales_value.sum().unstack(fill_value=0.0)
    gw = gw.reindex(keys).fillna(0.0)
    for k in range(14):
        if k not in gw.columns: gw[k] = 0.0
    gw = gw[sorted(gw.columns)]
    W = gw.loc[:, 0:12].values          # 13 trailing 28d windows, k=0 most recent
    spend_28 = gw[0]
    act = (W > 0).astype(float)
    wts = 2.0 ** (-(np.arange(13)) / 2.0)   # half-life 2 windows
    p13 = act @ wts / wts.sum()
    p6  = act[:, :6] @ wts[:6] / wts[:6].sum()
    p3  = act[:, :3] @ wts[:3] / wts[:3].sum()
    wa = wts[None, :] * act
    usual13 = (W * wa).sum(1) / np.maximum(wa.sum(1), 1e-9)
    wa6 = wts[None, :6] * act[:, :6]
    usual6 = (W[:, :6] * wa6).sum(1) / np.maximum(wa6.sum(1), 1e-9)
    wa3 = wts[None, :3] * act[:, :3]
    usual3 = (W[:, :3] * wa3).sum(1) / np.maximum(wa3.sum(1), 1e-9)
    # robust median over active windows
    umed = np.array([np.median(W[i][act[i] > 0]) if act[i].sum() > 0 else np.nan for i in range(W.shape[0])])
    umed = pd.Series(umed).fillna(pd.Series(usual13)).values
    streak = np.array([len(np.cumprod(act[i])) for i in range(W.shape[0])]).astype(float)
    # hazard: active->inactive transition rate over k=1..12
    num = ((act[:, 1:] > 0) & (act[:, :-1] == 0)).sum(1)
    den = (act[:, 1:] > 0).sum(1)
    hazard = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    cv13 = W.std(1) / (W.mean(1) + 1.0)
    # gaps within last 182d
    last = tx.groupby('household_key').day.max()
    dsl = (T - last).reindex(keys).fillna(730).values
    gap_long = np.full(len(keys), 999.0); gap_n21 = np.zeros(len(keys))
    g = tx[tx.day > T - 182].groupby('household_key').day.apply(lambda s: np.diff(np.sort(s.values))[::-1])
    for i, k_ in enumerate(keys):
        if k_ in g.index and len(g[k_]) > 0:
            gap_long[i] = g[k_].max(); gap_n21[i] = (g[k_] > 21).sum()
    ly_spend = gw[13]
    ly_ratio = np.clip(ly_spend / (spend_28 + 1.0), 0, 10)
    rvu = np.clip(spend_28 / (usual13 + 1.0), 0, 5)
    expected13 = p13 * usual13; expected6 = p6 * usual6; expected3 = p3 * usual3
    expected_med = p13 * umed
    exp_flat = W.mean(1)
    exp_rec = usual13 * np.exp(-dsl / 21.0)
    blend50 = 0.5 * expected13 + 0.5 * spend_28
    blend25 = 0.25 * expected13 + 0.75 * spend_28
    blend75 = 0.75 * expected13 + 0.25 * spend_28
    exp_streak = expected13 * np.clip(streak, 0, 4) / 4.0
    exp_rvu = expected13 * np.clip(rvu, 0, 3)
    out = pd.DataFrame({
        'spend_28': spend_28.values, 'p_active13': p13, 'p_active6': p6, 'p_act3': p3,
        'usual13': usual13, 'usual6': usual6, 'usual3': usual3, 'usual_med': umed,
        'expected13': expected13, 'expected6': expected6, 'expected3': expected3,
        'expected_med': expected_med, 'exp_flat': exp_flat, 'exp_rec': exp_rec,
        'blend50': blend50, 'blend25': blend25, 'blend75': blend75,
        'exp_streak': exp_streak, 'exp_rvu': exp_rvu,
        'streak': streak, 'hazard': hazard, 'cv13': cv13, 'gap_long': gap_long,
        'gap_n21': gap_n21, 'ly_spend': ly_spend.values, 'ly_ratio': ly_ratio.values,
        'recent_vs_usual': rvu.values, 'dsl': dsl,
        'log_expected': np.log1p(expected13), 'log_blend50': np.log1p(blend50),
        'log_spend28': np.log1p(spend_28.values),
    }, index=keys)
    return out

df = agent_api.build_features(feats)
print(df.shape, type(view := None))
tt = agent_api.train_targets()
m = df.merge(tt, on=['household_key', 'snapshot_day'])
y = m['future_spend_4w'].values
print('MAE predict-0:', np.abs(y).mean())
cands = [c for c in df.columns if c not in ('household_key','snapshot_day')]
for c in cands:
    v = m[c].fillna(0).values
    print(f'{c:16s} MAE {np.abs(y-v).mean():8.3f}  corr {np.corrcoef(y, v)[0,1]:.3f}')
# oracle blend of spend_28 and expected13
a, b = m['spend_28'].values, m['expected13'].values
for w in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    print(f'blend w={w}: MAE {np.abs(y-(w*a+(1-w)*b)).mean():.3f}')
