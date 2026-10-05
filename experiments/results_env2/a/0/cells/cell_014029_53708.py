
import pandas as pd, numpy as np, agent_api as A, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')

f = load_saved('feats_v4.parquet').copy()
tt = train_targets()
train_days = A.snapshot_days()['train']
feat_cols = [c for c in f.columns if c not in ('household_key','snapshot_day')]
for c in feat_cols:
    if not pd.api.types.is_numeric_dtype(f[c]):
        f[c] = pd.factorize(f[c])[0].astype('float32')

# market aggregates from capped view (valid for all train snapshots <=431)
snap = A.snapshot(459)
tx = snap.transactions
day_tot = tx.groupby('day').sales_value.sum().reindex(range(1,460), fill_value=0.0)
day_hh  = tx.groupby('day').household_key.nunique().reindex(range(1,460), fill_value=0.0)
def mkt(d, lo_off, hi_off):
    lo, hi = d+lo_off, d+hi_off
    if lo < 1: return np.nan
    return day_tot.loc[lo:hi].sum()
def mkt_hh(d, lo_off, hi_off):
    lo, hi = d+lo_off, d+hi_off
    if lo < 1: return np.nan
    return day_hh.loc[lo:hi].sum()

mk = {}
for d in sorted(set(train_days)) + [459,487,515,543]:
    m28, m84, m168 = mkt(d,-27,0), mkt(d,-83,0), mkt(d,-167,0)
    h28 = mkt_hh(d,-27,0)
    mk[d] = dict(mkt28=m28, mkt84=m84, mkt168=m168,
                 mkt_mom=m28/(m84/3.0), mkt_hh28=h28, mkt_sph28=m28/max(h28,1),
                 mkt_yago=mkt(d,-363,-336))
mkdf = pd.DataFrame(mk).T.reset_index().rename(columns={'index':'snapshot_day'})
for c in mkdf.columns:
    if c!='snapshot_day': mkdf[c]=mkdf[c].astype('float32')
print(mkdf.round(2).to_string())

# household year-ago lag (NaN when unavailable)
hh_day = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
def ylag_map(d):
    lo,hi = d-363, d-336
    if lo < 1: return pd.Series(dtype=float)
    return hh_day[(hh_day.day>=lo)&(hh_day.day<=hi)].groupby('household_key').sales_value.sum()
yl = {d: ylag_map(d) for d in sorted(set(train_days))+[459,487,515,543]}

def add_cols(df):
    df = df.merge(mkdf, on='snapshot_day', how='left')
    yl_col = []
    for h,d in zip(df.household_key, df.snapshot_day):
        s = yl[d]
        yl_col.append(s.get(h, np.nan) if len(s) else np.nan)
    df['ylag'] = np.array(yl_col, dtype=float)
    df['ylag_ratio'] = df.ylag / (df.lag1_spend + 1.0)
    return df

f_mkt = add_cols(f.copy())
mc = ['mkt28','mkt84','mkt168','mkt_mom','mkt_hh28','mkt_sph28','mkt_yago','ylag','ylag_ratio']
for c in mc:
    if not pd.api.types.is_numeric_dtype(f_mkt[c]): f_mkt[c] = f_mkt[c].astype('float32')
print('\nmkt feature NaN counts on train snaps:', f_mkt[f_mkt.snapshot_day.isin(train_days)][mc].isna().sum().to_dict())

tr  = f[f.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
trm = f_mkt[f_mkt.snapshot_day.isin(train_days)].merge(tt, on=['household_key','snapshot_day'])
def X(df, cols): return df[cols].values.astype(np.float32)
BASE = dict(max_depth=5, min_child_weight=40, learning_rate=0.08, n_estimators=400,
            subsample=0.9, colsample_bytree=0.9, tree_method='hist', n_jobs=8)
def fit_q(Xtr, ytr, u):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=u, **BASE)
    m.fit(Xtr, ytr); return m

def run_fold(df, cols, fit_max, eval_days):
    fi = df.snapshot_day <= fit_max; ei = df.snapshot_day.isin(eval_days)
    Xf, yf = X(df[fi], cols), df.future_spend_4w.values[fi]
    Xe = X(df[ei], cols)
    return fit_q(Xf, yf, 0.5).predict(Xe)

fc = feat_cols; fcm = feat_cols + mc
res = {}
for tag, df, cols in [('v4', tr, fc), ('v4+mkt', trm, fcm), ('v4+mkt-noylag', trm, feat_cols+mc[:7])]:
    r1 = run_fold(df, cols, 403, [431]); r2 = run_fold(df, cols, 375, [403,431])
    e431 = tr.snapshot_day==431; e2 = tr.snapshot_day.isin([403,431])
    a = np.abs(r1-tr.future_spend_4w.values[e431]).mean(); b = np.abs(r2-tr.future_spend_4w.values[e2]).mean()
    res[tag] = (a,b,(a+b)/2)
    print('%-16s f1 %.2f  f2 %.2f  avg %.2f' % (tag,a,b,(a+b)/2))
