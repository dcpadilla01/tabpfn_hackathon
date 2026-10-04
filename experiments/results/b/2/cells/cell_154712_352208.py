
import agent_api, pandas as pd, numpy as np

def cand_fn(view, snapshot_day):
    hh = view.households
    t = view.transactions
    d = t['day'].values; sv = t['sales_value'].values
    hk = pd.factorize(t['household_key'])[0]
    bid = pd.factorize(t['basket_id'].values)[0]
    n_hh = len(hh)
    out = {}
    def win_spend(mask_lo, mask_hi):
        m = (d > snapshot_day - mask_hi) & (d <= snapshot_day - mask_lo)
        return np.bincount(hk[m], weights=sv[m], minlength=n_hh)
    # trailing 28d spend at previous snapshot days (same-window lags)
    s28 = {}
    for k in (1,2,3,4,6):
        s28[k] = win_spend(28*k, 28*(k+1))
    stack = np.stack([s28[k] for k in (1,2,3,4,6)])
    out['c_hh_mean28'] = stack.mean(0)
    out['c_hh_med28'] = np.median(stack, 0)
    out['c_dev28'] = s28[1] / np.maximum(out['c_hh_mean28'], 1e-6)
    out['c_dev28_log'] = np.log1p(s28[1]) - np.log1p(out['c_hh_mean28'])
    # 84d windows at d-28, d-56
    s84a = win_spend(28, 112); s84b = win_spend(56, 140)
    out['c_hh_mean84'] = (s84a + s84b) / 2
    out['c_dev84'] = win_spend(0, 84) / np.maximum(out['c_hh_mean84'], 1e-6)
    # weekly CV over 26 weeks
    wk = (d - 1) // 7
    m26 = d > snapshot_day - 182
    wsum = pd.Series(sv[m26]).groupby(wk[m26]).sum()
    g = pd.Series(wsum).groupby(hk[m26][pd.Series(wsum.index.values).searchsorted(0)] if False else None)
    # simpler: per-household weekly sums via bincount on (hh, week) pairs
    wks = wk[m26]; hhk = hk[m26]; svs = sv[m26]
    pair = hhk * 100 + (wks - wks.min())
    tot = np.bincount(pair, weights=svs)
    nw = wks.max() - wks.min() + 1
    tot2 = tot.reshape(n_hh, nw) if len(tot) == n_hh*nw else None
    if tot2 is not None:
        mu = tot2.mean(1); sd = tot2.std(1)
        out['c_cv26'] = sd / np.maximum(mu, 1e-6)
    # burst: last-7d share of 28d
    out['c_burst7'] = win_spend(0,7) / np.maximum(win_spend(0,28), 1e-6)
    # distinct products & depts 84d
    m84 = d > snapshot_day - 84
    prod = t['product_id'].values[m84]
    dpair = hk[m84].astype(np.int64) * 1000003 + pd.factorize(prod)[0]
    out['c_nprod84'] = np.bincount(pd.unique(dpair).astype(np.int64) % (n_hh*1000003) // 1000003 if False else np.array([]), minlength=n_hh) if False else pd.Series(dpair).groupby(dpair).size().groupby(lambda x: x // 1000003).size().reindex(range(n_hh)).fillna(0).values
    # gap trend: mean inter-basket gap 28d vs 84d
    def gap_mean(lo, hi):
        m = (d > snapshot_day - hi) & (d <= snapshot_day - lo)
        df = pd.DataFrame({'h': hk[m], 'b': bid[m], 'dd': d[m]}).drop_duplicates(['h','b'])
        g = df.groupby('h')['dd'].agg(lambda x: np.mean(np.diff(np.sort(x.values))) if len(x) > 1 else np.nan)
        return g.reindex(range(n_hh)).values
    g28 = gap_mean(0, 28); g84 = gap_mean(0, 84)
    out['c_gap_trend'] = g28 / np.maximum(g84, 1e-6)
    # stock-up concentration: max basket 28d / spend_28
    m28 = d > snapshot_day - 28
    bs = pd.Series(sv[m28]).groupby(hk[m28] * 1000 + bid[m28]).sum()
    bmax = bs.groupby(lambda x: x // 1000).max().reindex(range(n_hh)).fillna(0).values
    out['c_conc28'] = bmax / np.maximum(win_spend(0,28), 1e-6)
    # coupon redemptions 84d
    cr = view.coupon_redemptions
    if cr is not None and len(cr):
        chr_ = pd.factorize(cr['household_key'])[0]
        cd = cr['day'].values
        mcr = cd > snapshot_day - 84
        out['c_red84'] = np.bincount(chr_[mcr], minlength=n_hh)
    else:
        out['c_red84'] = np.zeros(n_hh)
    # store count 28d
    st = t['store_id'].values[m28]
    spair = hk[m28].astype(np.int64) * 1000 + pd.factorize(st)[0]
    uniq = pd.unique(spair)
    out['c_nstore28'] = pd.Series(uniq // 1000).value_counts().reindex(range(n_hh)).fillna(0).values
    idx = pd.Index(hh, name='household_key')
    return pd.DataFrame(out, index=idx)

cand = agent_api.build_features(cand_fn)
print(cand.shape, cand.columns.tolist())
agent_api.save_table(cand, 'cand_screen1')
