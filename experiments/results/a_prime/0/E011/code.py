names = ['blocks_v1','e005_full_plus_mix','e006_curated','e006_temporal','e007_robust','e008_dist','e009_demo','e010_rhythm','hist_v1','hist_v2','mix_v1','mkt_v1','mkt_v2','rhythm_v1','season_v1','structure_v1']
for n in names:
    try:
        df = agent_api.load_saved(n + '.parquet')
        print('==', n, df.shape)
        print(df.columns.tolist())
    except Exception as e:
        print(n, 'ERR', repr(e))
tt = agent_api.train_targets()
print('targets', tt.shape)
print(tt.future_spend_4w.describe())
print('zero share', float((tt.future_spend_4w == 0).mean()))
v = agent_api.snapshot()
print('tx shape', v.transactions.shape)
print('n households at 459:', len(v.households))


# ---- cell ----
import numpy as np, pandas as pd
e10 = agent_api.load_saved('e010_rhythm.parquet')
print('e010 cols:', e10.columns.tolist())
tt = agent_api.train_targets()
m = e10.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
# per-snapshot drift of target
g = m.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median',lambda s:(s==0).mean()])
g.columns=['mean','median','zero_rate']
print(g)
# correlations with target
num = [c for c in e10.columns if c not in ('household_key','snapshot_day')]
cors = []
for c in num:
    s = m[c]
    if s.dtype==object or str(s.dtype)=='category':
        continue
    cc = s.corr(m.future_spend_4w)
    cors.append((c, cc))
cors = [(c,cc) for c,cc in cors if not np.isnan(cc)]
cors.sort(key=lambda t: abs(t[1]), reverse=True)
print('top |corr| with y:')
for c,cc in cors[:25]: print(f'  {c:24s} {cc: .3f}')
print('bottom:')
for c,cc in cors[-8:]: print(f'  {c:24s} {cc: .3f}')


# ---- cell ----
import numpy as np, pandas as pd

def probe(view):
    out = []
    out.append(('view.day', view.day))
    out.append(('view.week', view.week))
    out.append(('n_hh', len(view.households)))
    try:
        t = agent_api.train_targets()
        out.append(('train_targets', 'OK ' + str(t.shape)))
    except Exception as e:
        out.append(('train_targets', 'ERR ' + repr(e)[:90]))
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        out.append(('load_saved', 'OK ' + str(d.shape)))
    except Exception as e:
        out.append(('load_saved', 'ERR ' + repr(e)[:90]))
    try:
        h = agent_api.history(list(view.households)[:2], as_of_day=None)
        out.append(('history', 'OK ' + str(h.shape)))
    except Exception as e:
        out.append(('history', 'ERR ' + repr(e)[:90]))
    try:
        s2 = agent_api.snapshot(as_of_day=100)
        out.append(('snapshot(100)', 'OK tx ' + str(s2.transactions.shape)))
    except Exception as e:
        out.append(('snapshot(100)', 'ERR ' + repr(e)[:90]))
    for k, v in out:
        print(k, '->', v)
    # check view.transactions day range and view.households vs snapshot households
    print('tx day max', view.transactions.day.max(), 'min', view.transactions.day.min())
    print('demo rows', view.demographics.shape)
    print('camp rows', view.campaigns.shape, 'targets', view.campaign_targets.shape)
    print('redemptions', view.coupon_redemptions.shape)
    print('display_mailer', view.display_mailer.shape)
    return pd.DataFrame({'x': [1.0]}, index=list(view.households)[:1])

df = agent_api.build_features(probe)
print('built', df.shape)


# ---- cell ----
import numpy as np, pandas as pd

def probe(view, day):
    out = []
    out.append(('day', day, view.day, view.week))
    out.append(('n_hh', len(view.households), None, None))
    try:
        t = agent_api.train_targets()
        out.append(('train_targets', 'OK ' + str(t.shape), None, None))
    except Exception as e:
        out.append(('train_targets', 'ERR ' + repr(e)[:90], None, None))
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        out.append(('load_saved', 'OK ' + str(d.shape), None, None))
    except Exception as e:
        out.append(('load_saved', 'ERR ' + repr(e)[:90], None, None))
    try:
        h = agent_api.history(list(view.households)[:2], as_of_day=None)
        out.append(('history', 'OK ' + str(h.shape), None, None))
    except Exception as e:
        out.append(('history', 'ERR ' + repr(e)[:90], None, None))
    for row in out:
        print(row[0], '->', row[1])
    print('tx day max', view.transactions.day.max(), 'min', view.transactions.day.min())
    print('demo rows', view.demographics.shape)
    print('camp rows', view.campaigns.shape, 'targets', view.campaign_targets.shape)
    print('redemptions', view.coupon_redemptions.shape)
    print('display_mailer', view.display_mailer.shape)
    return pd.DataFrame({'x': [1.0]}, index=list(view.households)[:1])

df = agent_api.build_features(probe)
print('built', df.shape)


# ---- cell ----
import numpy as np, pandas as pd

def probe(view, day):
    diag = {}
    diag['day'] = day
    diag['view_day'] = view.day
    diag['view_week'] = view.week
    diag['n_hh'] = len(view.households)
    diag['tx_max_day'] = view.transactions.day.max()
    diag['tx_min_day'] = view.transactions.day.min()
    diag['demo_rows'] = view.demographics.shape[0]
    diag['camp_rows'] = view.campaigns.shape[0]
    diag['ct_rows'] = view.campaign_targets.shape[0]
    diag['red_rows'] = view.coupon_redemptions.shape[0]
    diag['dm_rows'] = view.display_mailer.shape[0]
    diag['dm_max_week'] = view.display_mailer.week_no.max()
    try:
        t = agent_api.train_targets()
        diag['tt'] = 'OK %d' % t.shape[0]
    except Exception as e:
        diag['tt'] = 'ERR ' + repr(e)[:80]
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        diag['ls'] = 'OK %d' % d.shape[0]
    except Exception as e:
        diag['ls'] = 'ERR ' + repr(e)[:80]
    try:
        h = agent_api.history(list(view.households)[:2])
        diag['hist'] = 'OK %d' % h.shape[0]
    except Exception as e:
        diag['hist'] = 'ERR ' + repr(e)[:80]
    return pd.DataFrame(diag, index=['h1'])

df = agent_api.build_features(probe)
print(df.T)


# ---- cell ----
import numpy as np, pandas as pd

def probe(view, day):
    diag = {}
    diag['day'] = day
    diag['view_day'] = view.day
    diag['view_week'] = view.week
    diag['n_hh'] = len(view.households)
    diag['tx_max_day'] = view.transactions.day.max()
    diag['demo_rows'] = view.demographics.shape[0]
    diag['camp_rows'] = view.campaigns.shape[0]
    diag['ct_rows'] = view.campaign_targets.shape[0]
    diag['red_rows'] = view.coupon_redemptions.shape[0]
    diag['dm_rows'] = view.display_mailer.shape[0]
    diag['dm_max_week'] = view.display_mailer.week_no.max()
    try:
        t = agent_api.train_targets()
        diag['tt'] = 'OK %d' % t.shape[0]
    except Exception as e:
        diag['tt'] = 'ERR ' + repr(e)[:80]
    try:
        d = agent_api.load_saved('mkt_v2.parquet')
        diag['ls'] = 'OK %d' % d.shape[0]
    except Exception as e:
        diag['ls'] = 'ERR ' + repr(e)[:80]
    try:
        h = agent_api.history(list(view.households)[:2])
        diag['hist'] = 'OK %d' % h.shape[0]
    except Exception as e:
        diag['hist'] = 'ERR ' + repr(e)[:80]
    return pd.DataFrame(diag, index=list(view.households)[:1])

df = agent_api.build_features(probe)
print(df.drop_duplicates(subset=['snapshot_day']).drop(columns=['household_key']).T)


# ---- cell ----
import numpy as np, pandas as pd

def rank_block(view, day):
    d = day
    hh = pd.Index(view.households)
    t = view.transactions[['household_key','basket_id','day','sales_value']]
    m28 = (t.day > d-28) & (t.day <= d)
    m84 = (t.day > d-84) & (t.day <= d)
    m364 = (t.day > d-364) & (t.day <= d)
    g28 = t[m28].groupby('household_key').agg(s28=('sales_value','sum'), b28=('basket_id','nunique'), days28=('day','nunique'))
    g84 = t[m84].groupby('household_key').agg(s84=('sales_value','sum'), b84=('basket_id','nunique'))
    g364 = t[m364].groupby('household_key').agg(s364=('sales_value','sum'))
    lastd = t.groupby('household_key')['day'].max()
    X = pd.DataFrame(index=hh)
    X = X.join(g28, how='left').join(g84, how='left').join(g364, how='left')
    for c, v in [('s28',0.0),('b28',0),('days28',0),('s84',0.0),('b84',0),('s364',0.0)]:
        X[c] = X[c].fillna(v)
    X['rec'] = (d - lastd).reindex(hh)
    X['rec'] = X['rec'].fillna(9999).astype(float)
    X['bv28'] = X.s28 / X.b28.replace(0, np.nan)
    X['trend'] = X.s28 / (X.s84 + 1.0)
    feats = {}
    for c in ['s28','s84','s364','b28','b84','days28','rec','bv28','trend']:
        feats['rk_'+c+'_pct'] = X[c].rank(pct=True)
    for c in ['s28','s84','b28','rec']:
        med = X[c].median(); iqr = X[c].quantile(0.75) - X[c].quantile(0.25)
        feats['rz_'+c] = ((X[c]-med)/(iqr+1e-9)).clip(-8, 8)
    feats['coh_median_s28'] = float(X.s28.median())
    feats['coh_mean_s28'] = float(X.s28.mean())
    feats['coh_iqr_s28'] = float(X.s28.quantile(0.75) - X.s28.quantile(0.25))
    feats['coh_active_rate'] = float((X.s28 > 0).mean())
    feats['coh_n'] = float(len(X))
    feats['coh_median_rec'] = float(X.rec.median())
    feats['coh_median_b28'] = float(X.b28.median())
    return pd.DataFrame(feats, index=hh)

df = agent_api.build_features(rank_block)
print('block built:', df.shape, df.columns.tolist())
e10 = agent_api.load_saved('e010_rhythm.parquet').drop(columns=['index'])
m = e10.merge(df, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)
p = agent_api.save_table(m, 'e011_rank.parquet')
print('saved:', p)
