import agent_api, pandas as pd, numpy as np

def feats(view, T):
    hh = view.households
    keys = pd.Index(pd.Series(hh).unique(), name='household_key')
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(keys)) & (tx.day > T - 730)].copy()
    tx['k'] = (T - tx.day) // 28
    gw = tx.groupby(['household_key','k']).sales_value.sum().unstack(fill_value=0.0)
    gw = gw.reindex(keys).fillna(0.0)
    for k in range(14):
        if k not in gw.columns: gw[k] = 0.0
    gw = gw[sorted(gw.columns)]
    W = gw.loc[:, 0:12].values
    spend_28 = gw[0]
    act = (W > 0).astype(float)
    def dec(hl): return 2.0 ** (-(np.arange(13)) / hl)
    w2, w4 = dec(2.0), dec(4.0)
    p13_2 = act @ w2 / w2.sum(); p13_4 = act @ w4 / w4.sum()
    p6_2  = act[:, :6] @ w2[:6] / w2[:6].sum(); p6_4 = act[:, :6] @ w4[:6] / w4[:6].sum()
    p3_2  = act[:, :3] @ w2[:3] / w2[:3].sum()
    def wmean(Wm, Am, w):
        wa = w[None, :] * Am
        return (Wm * wa).sum(1) / np.maximum(wa.sum(1), 1e-9)
    usual13 = wmean(W, act, w2); usual6 = wmean(W[:, :6], act[:, :6], w2[:6])
    usual13_4 = wmean(W, act, w4); usual6_4 = wmean(W[:, :6], act[:, :6], w4[:6])
    usual3 = wmean(W[:, :3], act[:, :3], w2[:3])
    umed = np.array([np.median(W[i][act[i] > 0]) if act[i].sum() > 0 else np.nan for i in range(W.shape[0])])
    umed = pd.Series(umed).fillna(pd.Series(usual13)).values
    streak = np.array([len(np.cumprod(act[i])) for i in range(W.shape[0])]).astype(float)
    num = ((act[:, 1:] > 0) & (act[:, :-1] == 0)).sum(1)
    den = (act[:, 1:] > 0).sum(1)
    hazard = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    cv13 = W.std(1) / (W.mean(1) + 1.0)
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
    e13 = p13_2 * usual13; e6 = p6_2 * usual6; e3 = p3_2 * usual3
    e13_4 = p13_4 * usual13_4; e6_4 = p6_4 * usual6_4
    e_med = p13_2 * umed
    e_rec = usual13 * np.exp(-dsl / 21.0)
    b75 = 0.75 * e6 + 0.25 * spend_28
    b50 = 0.5 * e6 + 0.5 * spend_28
    b75_13 = 0.75 * e13 + 0.25 * spend_28
    out = pd.DataFrame({
        'spend_28': spend_28.values, 'p13': p13_2, 'p6': p6_2, 'p3': p3_2,
        'p13_4': p13_4, 'p6_4': p6_4,
        'usual13': usual13, 'usual6': usual6, 'usual3': usual3,
        'usual13_4': usual13_4, 'usual6_4': usual6_4, 'usual_med': umed,
        'e13': e13, 'e6': e6, 'e3': e3, 'e13_4': e13_4, 'e6_4': e6_4,
        'e_med': e_med, 'e_rec': e_rec, 'b75': b75, 'b50': b50, 'b75_13': b75_13,
        'streak': streak, 'hazard': hazard, 'cv13': cv13, 'gap_long': gap_long,
        'gap_n21': gap_n21, 'ly_spend': ly_spend.values, 'ly_ratio': ly_ratio.values,
        'rvu': rvu.values, 'dsl': dsl,
        'log_e13': np.log1p(e13), 'log_e6': np.log1p(e6), 'log_b75': np.log1p(b75),
        'log_spend28': np.log1p(spend_28.values), 'log_usual13': np.log1p(usual13),
    }, index=keys)
    return out

df = agent_api.build_features(feats)
path = agent_api.save_table(df, 'e008_decomp2.parquet')
print(path, df.shape)
