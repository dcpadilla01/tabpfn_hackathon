
import agent_api
import pandas as pd, numpy as np

names = ['e001_recent_spend','e004_temporal','e005_longrun','e006_fwd_profile','e008_fwd_calendar',
         'e009_demo','e012_basket_shape','e009_demo_mkt','e011_discounts']
for n in names:
    try:
        t = agent_api.load_saved(n + '.parquet')
        print('==', n, t.shape)
        print(list(t.columns))
        print()
    except Exception as e:
        print(n, 'ERR', repr(e))

tt = agent_api.train_targets()
ycol = agent_api.TARGET
print('targets', tt.shape, tt.columns.tolist())
y = tt[ycol]
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2).to_string())
print('zero share', round(float((y==0).mean()),3))
print(tt.groupby('snapshot_day')[ycol].agg(['mean','median']).round(1).to_string())

t12 = agent_api.load_saved('e012_basket_shape.parquet')
m = t12.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
num = [c for c in m.columns if c not in ('household_key','snapshot_day',ycol) and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m[ycol])
print('--- correlations with target (sorted) ---')
print(cor.sort_values().round(3).to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

t12 = agent_api.load_saved('e012_basket_shape.parquet')
print(t12.shape)
print(list(t12.columns))
tt = agent_api.train_targets()
ycol = agent_api.TARGET
m = t12.merge(tt, on=['household_key','snapshot_day'], how='inner')
num = [c for c in m.columns if c not in ('household_key','snapshot_day',ycol) and pd.api.types.is_numeric_dtype(m[c])]
cor = m[num].corrwith(m[ycol]).sort_values()
print('--- 45 weakest correlations ---')
print(cor.head(45).round(3).to_string())
print('--- target dist ---')
y = tt[ycol]
print(y.describe(percentiles=[.5,.75,.9,.95,.99]).round(2).to_string())
print('zero share', round(float((y==0).mean()),3), 'mean', round(float(y.mean()),2), 'MAE of global median', round(float((y-y.median()).abs().mean()),2))
print('MAE of predict spend_28 (from e012):', )
m2 = m.dropna(subset=['spend_28'])
print(round(float((m2[ycol]-m2['spend_28']).abs().mean()),2))
print('MAE of 0.8*spend_28:', round(float((m2[ycol]-0.8*m2['spend_28']).abs().mean()),2))
print('snapdays', agent_api.snapshot_days())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def prep(t):
    return t.set_index(['household_key','snapshot_day'])

def ridge_cv(base, extra=None, use_log=True, alphas=(1,3,10,30,100,300), seed_split=(375,403,431)):
    frames = [prep(base)]
    if extra is not None: frames.append(prep(extra))
    df = pd.concat(frames, axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    feat = [c for c in df.columns if c != ycol]
    num = df[feat].select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float)
    X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    va_days = seed_split
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(va_days)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(itr.sum()), Xs[itr].values]; Xva = np.c_[np.ones(iva.sum()), Xs[iva].values]
    out = {}
    for tgt_mode in ([use_log] if use_log is not True else [True, False]):
        yy = np.log1p(y) if tgt_mode else y
        ytr = yy[itr].values; yva = y[iva].values
        best = None
        for a in alphas:
            A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
            w = np.linalg.solve(A, Xtr.T@ytr)
            p = np.clip(np.expm1(Xva@w) if tgt_mode else Xva@w, 0, None)
            mae = float(np.abs(p-yva).mean())
            if best is None or mae < best[0]: best = (mae, a)
        out['log' if tgt_mode else 'raw'] = (round(best[0],3), best[1])
    return out

base = agent_api.load_saved('e012_basket_shape.parquet')
print('base only:', ridge_cv(base))
# sanity: tiny base (spend_28 only)
tiny = base[['household_key','snapshot_day','spend_28','spend_84','wk_avg_4','fwd28_mean']]
print('tiny:', ridge_cv(tiny))


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def cand_fn(view, snapshot_day):
    hh = view.households
    t = view.transactions
    d = t['day'].values; sv = t['sales_value'].values
    hk_all = pd.factorize(t['household_key'].values)[0]
    n_all = hk_all.max() + 1
    bid = pd.factorize(t['basket_id'].values)[0]
    out = {}
    def wsum(lo, hi):
        m = (d > snapshot_day - hi) & (d <= snapshot_day - lo)
        return np.bincount(hk_all[m], weights=sv[m], minlength=n_all)
    s28 = {k: wsum(28*k, 28*(k+1)) for k in (1,2,3,4,6)}
    stack = np.stack([s28[k] for k in (1,2,3,4,6)])
    out['c_hh_mean28'] = stack.mean(0)
    out['c_hh_med28'] = np.median(stack, 0)
    out['c_dev28'] = s28[1] / np.maximum(out['c_hh_mean28'], 1e-6)
    out['c_dev28_log'] = np.log1p(s28[1]) - np.log1p(out['c_hh_mean28'])
    s84a = wsum(28, 112); s84b = wsum(56, 140)
    out['c_hh_mean84'] = (s84a + s84b) / 2
    out['c_dev84'] = wsum(0, 84) / np.maximum(out['c_hh_mean84'], 1e-6)
    # weekly CV over 26 weeks
    wk = (d - 1) // 7
    m26 = d > snapshot_day - 182
    wks = wk[m26]; hhk = hk_all[m26]; svs = sv[m26]
    if len(wks):
        wmin = wks.min(); off = wks - wmin; nw = int(off.max()) + 1
        pair = hhk.astype(np.int64) * nw + off
        tot = np.bincount(pair, weights=svs, minlength=n_all*nw).reshape(n_all, nw)
        mu = tot.mean(1); sd = tot.std(1)
        out['c_cv26'] = sd / np.maximum(mu, 1e-6)
    else:
        out['c_cv26'] = np.full(n_all, np.nan)
    out['c_burst7'] = wsum(0,7) / np.maximum(wsum(0,28), 1e-6)
    # distinct products 84d
    m84 = d > snapshot_day - 84
    prod = t['product_id'].values[m84]
    pcode = pd.factorize(prod)[0]
    dpair = hk_all[m84].astype(np.int64) * 1000000 + pcode
    upair = pd.unique(dpair)
    cnt = pd.Series(upair // 1000000).value_counts()
    out['c_nprod84'] = cnt.reindex(range(n_all)).fillna(0).values
    # gap trend
    def gap_mean(lo, hi):
        m = (d > snapshot_day - hi) & (d <= snapshot_day - lo)
        df = pd.DataFrame({'h': hk_all[m], 'b': bid[m], 'dd': d[m]}).drop_duplicates(['h','b'])
        g = df.groupby('h')['dd'].apply(lambda x: np.mean(np.diff(np.sort(x.values))) if len(x) > 1 else np.nan)
        return g.reindex(range(n_all)).values
    g28 = gap_mean(0, 28); g84 = gap_mean(0, 84)
    out['c_gap_trend'] = g28 / np.maximum(g84, 1e-6)
    # stock-up concentration
    m28 = d > snapshot_day - 28
    bs = pd.Series(sv[m28]).groupby(hk_all[m28].astype(np.int64)*100000 + bid[m28]).sum()
    bmax = bs.groupby(bs.index // 100000).max().reindex(range(n_all)).fillna(0).values
    out['c_conc28'] = bmax / np.maximum(wsum(0,28), 1e-6)
    cr = view.coupon_redemptions
    if cr is not None and len(cr):
        chr_ = pd.factorize(cr['household_key'].values)[0]
        mcr = cr['day'].values > snapshot_day - 84
        out['c_red84'] = np.bincount(chr_[mcr], minlength=n_all)
    else:
        out['c_red84'] = np.zeros(n_all)
    st = t['store_id'].values[m28]
    scode = pd.factorize(st)[0]
    spair = hk_all[m28].astype(np.int64) * 10000 + scode
    us = pd.unique(spair)
    out['c_nstore28'] = pd.Series(us // 10000).value_counts().reindex(range(n_all)).fillna(0).values
    idx = pd.Index(hh, name='household_key')
    return pd.DataFrame(out, index=idx)

cand = agent_api.build_features(cand_fn)
print(cand.shape, cand.columns.tolist())
print(cand.isna().mean().round(3).to_string())
agent_api.save_table(cand, 'cand_screen1')


# ---- cell ----

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


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, alphas=(1,3,10,30,100,300,1000), seed_split=(375,403,431)):
    df = pd.concat([f.set_index(['household_key','snapshot_day']) for f in frames], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(seed_split)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(itr.sum()), Xs[itr].values]; Xva = np.c_[np.ones(iva.sum()), Xs[iva].values]
    ytr = np.log1p(y[itr]).values; yva = y[iva].values
    res = []
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(np.expm1(Xva@w), 0, None)
        res.append((float(np.abs(p-yva).mean()), a))
    res.sort()
    return round(res[0][0],3), res[0][1], round(res[-1][0],3)

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
print('base           :', ridge_cv([base]))
print('base + cand    :', ridge_cv([base, cand]))
# ablation: drop one cand feature at a time
ccols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
full = ridge_cv([base, cand])[0]
print('full:', full)
for c in ccols:
    sub = cand.drop(columns=[c])
    m, a, worst = ridge_cv([base, sub])
    print(f'drop {c:14s}: {m}  (delta {round(m-full,3):+.3f})')


# ---- cell ----

import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
ycol = agent_api.TARGET

def prep_df(t):
    df = t.set_index(['household_key','snapshot_day']).join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    return X, y

X, y = prep_df(base)
seeds = (375,403,431)
tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seeds]
itr = X.index.get_level_values(1).isin(tr_days).values
iva = X.index.get_level_values(1).isin(seeds).values
print('n train rows', itr.sum(), 'n seed rows', iva.sum())
mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
Xs = ((X-mu)/sd).values
Xtr = np.c_[np.ones(itr.sum()), Xs[itr]]; Xva = np.c_[np.ones(iva.sum()), Xs[iva]]
ytr = np.log1p(y[itr]).values; yva = y[iva].values
for a in (1,3,10,30,100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr)
    p = np.clip(np.expm1(Xva@w), 0, None)
    print(f'alpha={a:5d}  logMAE={np.abs(p-yva).mean():.3f}')
# also raw-target variant
ytr2 = y[itr].values
for a in (100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr2)
    p = np.clip(Xva@w, 0, None)
    print(f'alpha={a:5d}  rawMAE={np.abs(p-yva).mean():.3f}')


# ---- cell ----

import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
ycol = agent_api.TARGET

df = base.set_index(['household_key','snapshot_day']).join(tt[ycol])
num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
X = df[num].astype(float); X = X.fillna(X.median())
y = df[ycol].astype(float)
seeds = (375,403,431)
tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seeds]
itr = X.index.get_level_values(1).isin(tr_days)
iva = X.index.get_level_values(1).isin(seeds)
print('n train rows', int(itr.sum()), 'n seed rows', int(iva.sum()))
mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
Xs = ((X-mu)/sd).values
Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr]]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva]]
ytr = np.log1p(y[itr]).values; yva = y[iva].values
for a in (1,3,10,30,100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr)
    p = np.clip(np.expm1(Xva@w), 0, None)
    print(f'alpha={a:5d}  logMAE={np.abs(p-yva).mean():.3f}')
ytr2 = y[itr].values
for a in (100,300,1000):
    A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
    w = np.linalg.solve(A, Xtr.T@ytr2)
    p = np.clip(Xva@w, 0, None)
    print(f'alpha={a:5d}  rawMAE={np.abs(p-yva).mean():.3f}')


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def ridge_cv_raw(frames, alphas=(300,1000,3000,10000), seed_split=(375,403,431)):
    df = pd.concat([f.set_index(['household_key','snapshot_day']) for f in frames], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(seed_split)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
    ytr = y[itr].values; yva = y[iva].values
    res = []
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(Xva@w, 0, None)
        res.append((float(np.abs(p-yva).mean()), a))
    res.sort()
    return res[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
b = ridge_cv_raw([base]); print('base raw:', b)
bc = ridge_cv_raw([base, cand]); print('base+cand raw:', bc)
full = bc[0]
ccols = [c for c in cand.columns if c not in ('household_key','snapshot_day')]
for c in ccols:
    m, a = ridge_cv_raw([base, cand.drop(columns=[c])])
    print(f'drop {c:14s}: {m:.3f}  (delta {m-full:+.3f})')
# cand alone
print('cand alone raw:', ridge_cv_raw([cand])[0])


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, mode='raw', alphas=(100,300,1000,3000,10000), seed_split=(375,403,431)):
    df = pd.concat([f.set_index(['household_key','snapshot_day']) for f in frames], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(seed_split)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
    ytr = (np.log1p(y[itr]) if mode=='log' else y[itr]).values; yva = y[iva].values
    best = None
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(np.expm1(Xva@w) if mode=='log' else Xva@w, 0, None)
        m = float(np.abs(p-yva).mean())
        if best is None or m < best[0]: best = (m, a)
    return best[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
b = base.set_index(['household_key','snapshot_day'])
print('base raw:', round(ridge_cv([base],'raw'),3), ' log:', round(ridge_cv([base],'log'),3))

# nonlinear block computed offline
nb = pd.DataFrame(index=b.index)
s28 = b['spend_28']; s84 = b['spend_84']; tr = b['trips_28']; rec = b['recency']
fwm = b['fwd28_mean']; wk4 = b['wk_avg_4']
nb['n_sq28'] = (s28/100)**2
nb['n_sqrt28'] = np.sqrt(np.clip(s28,0,None))
nb['n_rank28'] = s28.groupby(level=1).rank(pct=True)
nb['n_rank84'] = s84.groupby(level=1).rank(pct=True)
nb['n_rankfwm'] = fwm.groupby(level=1).rank(pct=True)
nb['n_int_st'] = s28*tr/100
nb['n_int_sr'] = s28*np.clip(rec,0,84)/100
nb['n_int_s84t'] = s84*tr/100
nb['n_log_sq'] = np.log1p(s28)**2
nb['n_ranktr'] = tr.groupby(level=1).rank(pct=True)
nbf = nb.reset_index()
print('base+nonlin raw:', round(ridge_cv([base,nbf],'raw'),3), ' log:', round(ridge_cv([base,nbf],'log'),3))
# ablation within nonlinear block
fullr = ridge_cv([base,nbf],'raw'); fulll = ridge_cv([base,nbf],'log')
for c in nb.columns:
    sub = nbf.drop(columns=[c])
    print(f"drop {c:12s} raw {ridge_cv([base,sub],'raw'):.3f} ({ridge_cv([base,sub],'raw')-fullr:+.3f})  log {ridge_cv([base,sub],'log'):.3f} ({ridge_cv([base,sub],'log')-fulll:+.3f})")
# nonlin alone on top of nothing? and smaller variants
print('nonlin alone raw:', round(ridge_cv([nbf],'raw'),3))


# ---- cell ----

import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
ycol = agent_api.TARGET
df = base.set_index(['household_key','snapshot_day']).join(tt[ycol])
num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
X = df[num].astype(float); X = X.fillna(X.median())
y = df[ycol].astype(float)
itr = X.index.get_level_values(1).isin([d for d in agent_api.snapshot_days()['train'] if d != 431])
iva = X.index.get_level_values(1).isin([431])
mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
Xs = (X-mu)/sd
Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
w = np.linalg.solve(Xtr.T@Xtr + 3000*np.eye(Xtr.shape[1]), Xtr.T@y[itr].values)
p = np.clip(Xva@w, 0, None)
res = y[iva].values - p
d431 = df[iva].copy(); d431['pred'] = p; d431['res'] = res; d431['y'] = y[iva].values
print('431 MAE', round(float(np.abs(res).mean()),2))
# residual by target bucket
d431['ybucket'] = pd.cut(d431['y'], [-1,0.01,50,100,200,400,800,10000])
print(d431.groupby('ybucket', observed=True).apply(lambda g: pd.Series({'n':len(g),'y':g['y'].mean(),'pred':g['pred'].mean(),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
# by has_demo / size
print(d431.groupby('has_demo').apply(lambda g: pd.Series({'n':len(g),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
print(d431.groupby('size_ord', dropna=False).apply(lambda g: pd.Series({'n':len(g),'mae':g['res'].abs().mean(),'bias':g['res'].mean()})).round(1).to_string())
# drift: target mean by snapshot day (train)
g = tt.groupby('snapshot_day')[ycol].agg(['mean','median','count'])
print(g.round(1).to_string())
# also mean of spend_28 by snapshot day (proxy for level drift)
s = base.groupby('snapshot_day')[['spend_28','wk_avg_8','fwd28_mean']].mean().round(1)
print(s.to_string())


# ---- cell ----

import agent_api, pandas as pd, numpy as np

def ridge_cv(frames, alphas=(300,1000,3000,10000), seed_split=(375,403,431)):
    df = pd.concat([f.set_index(['household_key','snapshot_day']) for f in frames], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    tt = agent_api.train_targets().set_index(['household_key','snapshot_day'])
    ycol = agent_api.TARGET
    df = df.join(tt[ycol])
    num = df.drop(columns=[ycol]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[num].astype(float); X = X.fillna(X.median())
    y = df[ycol].astype(float)
    tr_days = [d for d in agent_api.snapshot_days()['train'] if d not in seed_split]
    itr = df.index.get_level_values(1).isin(tr_days); iva = df.index.get_level_values(1).isin(seed_split)
    mu, sd = X[itr].mean(), X[itr].std().replace(0,1)
    Xs = (X-mu)/sd
    Xtr = np.c_[np.ones(int(itr.sum())), Xs[itr].values]; Xva = np.c_[np.ones(int(iva.sum())), Xs[iva].values]
    ytr = y[itr].values; yva = y[iva].values
    best = None
    for a in alphas:
        A = Xtr.T@Xtr + a*np.eye(Xtr.shape[1]); A[0,0] -= a
        w = np.linalg.solve(A, Xtr.T@ytr)
        p = np.clip(Xva@w, 0, None)
        m = float(np.abs(p-yva).mean())
        if best is None or m < best[0]: best = (m, a)
    return best[0]

base = agent_api.load_saved('e012_basket_shape.parquet')
b = base.set_index(['household_key','snapshot_day'])
print('base:', round(ridge_cv([base]),3))

# Block A: zero / tail targeting
A = pd.DataFrame(index=b.index)
rec = b['recency']; gm = b['gap_mean_84']; zf = b['zero_frac_13']; s28 = b['spend_28']; w8a = b['wk_avg_8']
A['z_rec45'] = (rec > 45).astype(float)
A['z_rec30'] = (rec > 30).astype(float)
A['z_gap30'] = (gm > 30).astype(float)
A['z_zf50']  = (zf > 0.5).astype(float)
A['z_rec45_s28'] = A['z_rec45']*s28
A['z_rec45_w8']  = A['z_rec45']*w8a
A['t_maxwk'] = b[['w3','w4','w5','w6','w7','w8']].max(axis=1)
A['t_p90wk'] = b[['w3','w4','w5','w6','w7','w8']].quantile(0.9, axis=1)
A['t_top2']  = (b['w7']+b['w8'])/np.maximum(b[['w3','w4','w5','w6','w7','w8']].sum(axis=1),1e-6)
A['t_s28_inc'] = s28*(1+b['income_ord'].fillna(0)*0)  # placeholder skip
Af = A.reset_index()
print('base+A:', round(ridge_cv([base,Af]),3))
# Block C: tenure-gated / shrunk rates
C = pd.DataFrame(index=b.index)
ten = b['tenure'].clip(lower=0)
C['c_gate28'] = s28*ten/(ten+50)
C['c_gate84'] = b['spend_84']*ten/(ten+100)
C['c_shr28'] = (s28*ten + 130*50)/(ten+50)   # blend with global prior ~130/4wk
C['c_shr84'] = (b['spend_84']*ten + 130*200)/(ten+200)
C['c_lograte'] = np.log1p(b['spend_rate_life'])
Cf = C.reset_index()
print('base+C:', round(ridge_cv([base,Cf]),3))
print('base+A+C:', round(ridge_cv([base,Af,Cf]),3))
# ablations
full = ridge_cv([base,Af,Cf])
for c in list(A.columns)+list(C.columns):
    sub = pd.concat([Af,Cf],axis=1).drop(columns=[c])
    m = ridge_cv([base,sub])
    print(f'drop {c:12s}: {m:.3f} ({m-full:+.3f})')


# ---- cell ----

import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
b = base.set_index(['household_key','snapshot_day'])

# curated blocks
keep_cand = ['c_dev28','c_dev84','c_cv26','c_burst7','c_gap_trend']
A = pd.DataFrame(index=b.index)
rec = b['recency']; gm = b['gap_mean_84']; zf = b['zero_frac_13']; s28 = b['spend_28']; w8a = b['wk_avg_8']
A['z_rec45'] = (rec > 45).astype(float)
A['z_gap30'] = (gm > 30).astype(float)
A['z_zf50']  = (zf > 0.5).astype(float)
wks = b[['w3','w4','w5','w6','w7','w8']]
A['t_maxwk'] = wks.max(axis=1)
A['t_p90wk'] = wks.quantile(0.9, axis=1)
A['t_top2']  = (b['w7']+b['w8'])/np.maximum(wks.sum(axis=1),1e-6)
C = pd.DataFrame(index=b.index)
ten = b['tenure'].clip(lower=0)
C['c_gate28'] = s28*ten/(ten+50)
C['c_gate84'] = b['spend_84']*ten/(ten+100)
C['c_shr28'] = (s28*ten + 130*50)/(ten+50)
C['c_shr84'] = (b['spend_84']*ten + 130*200)/(ten+200)
C['c_lograte'] = np.log1p(b['spend_rate_life'])
N = pd.DataFrame(index=b.index)
s84 = b['spend_84']; tr = b['trips_28']; fwm = b['fwd28_mean']
N['n_rank84'] = s84.groupby(level=1).rank(pct=True)
N['n_rank28'] = s28.groupby(level=1).rank(pct=True)
N['n_rankfwm'] = fwm.groupby(level=1).rank(pct=True)
N['n_ranktr'] = tr.groupby(level=1).rank(pct=True)
N['n_sqrt28'] = np.sqrt(np.clip(s28,0,None))
N['n_log_sq'] = np.log1p(s28)**2

blocks = [base,
          cand[keep_cand].reset_index(),
          A.reset_index(), C.reset_index(), N.reset_index()]
out = blocks[0]
for blk in blocks[1:]:
    out = out.merge(blk, on=['household_key','snapshot_day'], how='inner')
print(out.shape)
assert out.shape[0] == 36426
newcols = [c for c in out.columns if c not in base.columns]
print(len(newcols), newcols)
agent_api.save_table(out, 'e013_denoise')


# ---- cell ----

import agent_api, pandas as pd, numpy as np
base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
b = base.set_index(['household_key','snapshot_day'])
keep_cand = ['c_dev28','c_dev84','c_cv26','c_burst7','c_gap_trend']
A = pd.DataFrame(index=b.index)
rec = b['recency']; gm = b['gap_mean_84']; zf = b['zero_frac_13']; s28 = b['spend_28']; w8a = b['wk_avg_8']
A['z_rec45'] = (rec > 45).astype(float); A['z_gap30'] = (gm > 30).astype(float); A['z_zf50'] = (zf > 0.5).astype(float)
wks = b[['w3','w4','w5','w6','w7','w8']]
A['t_maxwk'] = wks.max(axis=1); A['t_p90wk'] = wks.quantile(0.9, axis=1)
A['t_top2'] = (b['w7']+b['w8'])/np.maximum(wks.sum(axis=1),1e-6)
C = pd.DataFrame(index=b.index)
ten = b['tenure'].clip(lower=0)
C['c_gate28'] = s28*ten/(ten+50); C['c_gate84'] = b['spend_84']*ten/(ten+100)
C['c_shr28'] = (s28*ten + 130*50)/(ten+50); C['c_shr84'] = (b['spend_84']*ten + 130*200)/(ten+200)
C['c_lograte'] = np.log1p(b['spend_rate_life'])
N = pd.DataFrame(index=b.index)
s84 = b['spend_84']; tr = b['trips_28']; fwm = b['fwd28_mean']
N['n_rank84'] = s84.groupby(level=1).rank(pct=True); N['n_rank28'] = s28.groupby(level=1).rank(pct=True)
N['n_rankfwm'] = fwm.groupby(level=1).rank(pct=True); N['n_ranktr'] = tr.groupby(level=1).rank(pct=True)
N['n_sqrt28'] = np.sqrt(np.clip(s28,0,None)); N['n_log_sq'] = np.log1p(s28)**2
out = base.merge(cand[['household_key','snapshot_day']+keep_cand], on=['household_key','snapshot_day'], how='inner')
for blk in (A, C, N):
    out = out.merge(blk.reset_index(), on=['household_key','snapshot_day'], how='inner')
print(out.shape)
assert out.shape[0] == 36426
newcols = [c for c in out.columns if c not in base.columns]
print(len(newcols), newcols)
agent_api.save_table(out, 'e013_denoise')
