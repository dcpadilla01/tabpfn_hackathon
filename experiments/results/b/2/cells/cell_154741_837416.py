
import agent_api, pandas as pd, numpy as np

def cand_fn(view, snapshot_day):
    hh = view.households
    t = view.transactions
    d = t['day'].values; sv = t['sales_value'].values
    codes, uniques = pd.factorize(t['household_key'].values)
    n_all = len(uniques)
    bid = pd.factorize(t['basket_id'].values)[0]
    out = {}
    def wsum(lo, hi):
        m = (d > snapshot_day - hi) & (d <= snapshot_day - lo)
        return np.bincount(codes[m], weights=sv[m], minlength=n_all)
    s28 = {k: wsum(28*k, 28*(k+1)) for k in (1,2,3,4,6)}
    stack = np.stack([s28[k] for k in (1,2,3,4,6)])
    out['c_hh_mean28'] = stack.mean(0)
    out['c_hh_med28'] = np.median(stack, 0)
    out['c_dev28'] = s28[1] / np.maximum(out['c_hh_mean28'], 1e-6)
    out['c_dev28_log'] = np.log1p(s28[1]) - np.log1p(out['c_hh_mean28'])
    s84a = wsum(28, 112); s84b = wsum(56, 140)
    out['c_hh_mean84'] = (s84a + s84b) / 2
    out['c_dev84'] = wsum(0, 84) / np.maximum(out['c_hh_mean84'], 1e-6)
    wk = (d - 1) // 7
    m26 = d > snapshot_day - 182
    wks = wk[m26]; hhk = codes[m26]; svs = sv[m26]
    if len(wks):
        wmin = wks.min(); off = wks - wmin; nw = int(off.max()) + 1
        pair = hhk.astype(np.int64) * nw + off
        tot = np.bincount(pair, weights=svs, minlength=n_all*nw).reshape(n_all, nw)
        mu = tot.mean(1); sd = tot.std(1)
        out['c_cv26'] = sd / np.maximum(mu, 1e-6)
    else:
        out['c_cv26'] = np.full(n_all, np.nan)
    out['c_burst7'] = wsum(0,7) / np.maximum(wsum(0,28), 1e-6)
    m84 = d > snapshot_day - 84
    pcode = pd.factorize(t['product_id'].values[m84])[0]
    dpair = codes[m84].astype(np.int64) * 1000000 + pcode
    upair = pd.unique(dpair)
    cnt = pd.Series(upair // 1000000).value_counts()
    out['c_nprod84'] = cnt.reindex(range(n_all)).fillna(0).values
    def gap_mean(lo, hi):
        m = (d > snapshot_day - hi) & (d <= snapshot_day - lo)
        df = pd.DataFrame({'h': codes[m], 'b': bid[m], 'dd': d[m]}).drop_duplicates(['h','b'])
        g = df.groupby('h')['dd'].apply(lambda x: np.mean(np.diff(np.sort(x.values))) if len(x) > 1 else np.nan)
        return g.reindex(range(n_all)).values
    g28 = gap_mean(0, 28); g84 = gap_mean(0, 84)
    out['c_gap_trend'] = g28 / np.maximum(g84, 1e-6)
    m28 = d > snapshot_day - 28
    bs = pd.Series(sv[m28]).groupby(codes[m28].astype(np.int64)*100000 + bid[m28]).sum()
    bmax = bs.groupby(bs.index // 100000).max().reindex(range(n_all)).fillna(0).values
    out['c_conc28'] = bmax / np.maximum(wsum(0,28), 1e-6)
    cr = view.coupon_redemptions
    if cr is not None and len(cr):
        ccode, _ = pd.factorize(cr['household_key'].values)
        mcr = cr['day'].values > snapshot_day - 84
        out['c_red84'] = np.bincount(ccode[mcr], minlength=n_all)
    else:
        out['c_red84'] = np.zeros(n_all)
    scode = pd.factorize(t['store_id'].values[m28])[0]
    spair = codes[m28].astype(np.int64) * 10000 + scode
    us = pd.unique(spair)
    out['c_nstore28'] = pd.Series(us // 10000).value_counts().reindex(range(n_all)).fillna(0).values
    df = pd.DataFrame(out, index=pd.Index(uniques, name='household_key'))
    return df.reindex(hh)

cand = agent_api.build_features(cand_fn)
print(cand.shape, cand.columns.tolist())
print(cand.isna().mean().round(3).to_string())
agent_api.save_table(cand, 'cand_screen1')
