api = agent_api
for name in ['e003_catmix.parquet','e001_txhist.parquet']:
    t = api.load_saved(name)
    print(name, t.shape)
    print(list(t.columns))
    print()
b = api.baseline_features()
print('baseline', b.shape)
print(list(b.columns))
tt = api.train_targets()
y = tt[api.TARGET]
print(y.describe())
print('zero share %.3f' % (y==0).mean())
print('quantiles', y.quantile([.5,.75,.9,.95,.99]).to_dict())

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api
t = api.load_saved('e003_catmix.parquet')
tt = api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='left')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
tr = df.snapshot_day <= 431; va = df.snapshot_day >= 459
Xtr = df.loc[tr, feats].astype(float).values
mu = np.nanmean(Xtr, 0); Xtr = np.where(np.isnan(Xtr), mu, Xtr)
ytr = df.loc[tr,'future_spend_4w'].values
Xva = df.loc[va, feats].astype(float).values; Xva = np.where(np.isnan(Xva), mu, Xva)
yva = df.loc[va,'future_spend_4w'].values
sd = Xtr.std(0); sd[sd==0]=1
Ztr=(Xtr-mu)/sd; Zva=(Xva-mu)/sd
def fit(lam):
    return np.linalg.solve(Ztr.T@Ztr + lam*np.eye(len(feats)), Ztr.T@ytr)
print('median-only MAE %.3f' % np.abs(np.median(ytr)-yva).mean())
best=None
for lam in [3,10,30,100,300,1000]:
    pv = Zva@fit(lam)
    m = np.abs(pv-yva).mean()
    print('ridge lam %g MAE %.3f' % (lam, m))
w = fit(100); pv = Zva@w; res = yva - pv
dva = df.loc[va].copy(); dva['pred']=pv; dva['res']=res; dva['y']=yva
print('\nBy true-y bucket:')
dva['yb'] = pd.cut(dva.y, [-1,0.5,50,150,300,700,1e9])
print(dva.groupby('yb', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'mean_pred':g.pred.mean(),'mean_y':g.y.mean()})))
print('\nBy tenure bucket:')
dva['tb'] = pd.cut(dva.tenure, [0,120,300,600,1e9])
print(dva.groupby('tb', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean()})))
print('\nBy days_since_last:')
dva['db'] = pd.cut(dva.days_since_last, [-1,0,7,14,28,60,1e9])
print(dva.groupby('db', observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'bias':g.res.mean()})))
print('\ncorr(pred,y)=%.3f' % np.corrcoef(pv,yva)[0,1])

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api
t = api.load_saved('e003_catmix.parquet')
tt = api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
days = sorted(df.snapshot_day.unique())
print('snapshots', days, 'rows', len(df))
def eval_laso(feats, lam=100, log=False):
    mae=[]; res_all=[]; df_all=[]
    for d in days:
        tr = df.snapshot_day!=d; va = df.snapshot_day==d
        Xtr = df.loc[tr, feats].astype(float).values
        mu = np.nanmean(Xtr,0); Xtr=np.where(np.isnan(Xtr),mu,Xtr)
        ytr = df.loc[tr,'future_spend_4w'].values
        Xva = df.loc[va, feats].astype(float).values; Xva=np.where(np.isnan(Xva),mu,Xva)
        yva = df.loc[va,'future_spend_4w'].values
        sd=Xtr.std(0); sd[sd==0]=1
        Ztr=(Xtr-mu)/sd; Zva=(Xva-mu)/sd
        if log:
            ytr=np.log1p(ytr)
        w=np.linalg.solve(Ztr.T@Ztr+lam*np.eye(len(feats)), Ztr.T@ytr)
        pv=Zva@w
        if log: pv=np.expm1(pv)
        r=pv-yva; mae.append(np.abs(r).mean()); res_all.append(r); df_all.append(df.loc[va].assign(pred=pv, res=r, y=yva))
    return np.mean(mae), pd.concat(df_all)
m,_=eval_laso(feats); print('E003 proxy MAE %.3f'%m)
m2,_=eval_laso(feats, log=True); print('E003 log-target proxy MAE %.3f'%m2)
_,D=eval_laso(feats)
D['yb']=pd.cut(D.y,[-1,.5,50,150,300,700,1e9])
print(D.groupby('yb',observed=True).apply(lambda g: pd.Series({'n':len(g),'mae':np.abs(g.res).mean(),'bias':g.res.mean(),'mp':g.pred.mean()})))
print('\nby snapshot:'); print(D.groupby('snapshot_day').apply(lambda g: np.abs(g.res).mean()))

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api
s = api.snapshot()  # capped at 459
tx = s.transactions
wk = tx.groupby('week_no').sales_value.sum()
hh = tx.groupby('week_no').household_key.nunique()
per_hh = (wk/hh)
print('weeks', wk.index.min(), wk.index.max())
print('weekly total spend: first 20 weeks:'); print(wk.head(20).round(0).to_dict())
print('per-household weekly spend, weeks 1..52:'); print(per_hh.head(52).round(1).to_dict())
print('overall per-hh mean %.1f std %.1f' % (per_hh.mean(), per_hh.std()))
# same-window-last-year anchor feasibility: for snapshot d, mean per-household spend in [d-27,d] and [d-363,d-336]
days = api.snapshot_days()['train']+api.snapshot_days()['validation']
print('val days', api.snapshot_days()['validation'])
tx2 = tx[['household_key','day','sales_value']]
for d in [95, 207, 431, 459]:
    w0 = tx2[(tx2.day>=d-27)&(tx2.day<=d)].groupby('household_key').sales_value.sum()
    w1 = tx2[(tx2.day>=d-363)&(tx2.day<=d-336)].groupby('household_key').sales_value.sum()
    print(d, 'recent4w mean %.1f  lastyear4w mean %.1f  n_ly %d' % (w0.mean(), w1.mean(), len(w1)))

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api

# --- candidate 1: demographics on top of E003 ---
e3 = api.load_saved('e003_catmix.parquet')
b = api.baseline_features()
demo_cols = ['classification_1','classification_2','classification_3','classification_4','classification_5','homeowner_desc','kid_category_desc','has_demographics']
d1 = e3.merge(b[['household_key','snapshot_day']+demo_cols], on=['household_key','snapshot_day'], how='left')
print('c1', d1.shape)
api.save_table(d1, 'e011_demo.parquet')

# --- candidate 2: dormancy / gap-structure features on top of E003 ---
def dorm(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    g = w.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # trip-day gaps per household
    trips = w.groupby(['household_key','day']).size().reset_index()[['household_key','day']]
    trips = trips.sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key').day.diff()
    gg = trips.groupby('household_key').gap
    out['gap_mean84'] = gg.mean()
    out['gap_max84'] = gg.max()
    out['gap_std84'] = gg.std()
    out['gap_last'] = trips.groupby('household_key').gap.last()
    # zero 28d windows among l1..l6
    sp = tx.groupby('household_key').day.apply(lambda s: None)  # placeholder
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    def win_zero(lo, hi):
        s = spend[(spend.day > d-hi) & (spend.day <= d-lo+0)].groupby('household_key').sales_value.sum()
        return s
    z = 0; cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        s = spend[(spend.day > d-28*(k+1)) & (spend.day <= d-28*k)].groupby('household_key').sales_value.sum()
        cnt = cnt + (s.reindex(hh).fillna(0) <= 0).astype(float)
    out['n_zero_l6'] = cnt
    # longest inactive run in last 84d (max gap between consecutive trip days, incl. edges)
    first = trips.groupby('household_key').day.min(); last = trips.groupby('household_key').day.max()
    out['edge_gap84'] = (last - first).where(first.notna())
    out['inactive_run84'] = np.maximum(out['gap_max84'].fillna(84), (d - last).where(last.notna(), 84))
    # full-tenure zero-window fraction
    allsp = tx.groupby('household_key').sales_value.sum()
    ten = tx.groupby('household_key').day.agg(['min','max'])
    out['tenure'] = (d - ten['min']).clip(lower=1)
    out['zero_frac_full'] = cnt / 6.0  # cheap proxy; refine below with full history
    # full-history 28d windows with zero spend
    fh = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    nw = (out['tenure'] // 28).clip(lower=1)
    # count zero windows over full tenure via weekly activity
    wk_act = tx.assign(wk=(tx.day+8)//7).groupby(['household_key','wk']).size()
    n_act_weeks = wk_act.groupby('household_key').size()
    n_weeks_ten = ((d - ten['min']) // 7).clip(lower=1)
    out['active_week_share'] = (n_act_weeks / n_weeks_ten).clip(0,1)
    return out

tab2 = api.build_features(lambda view, sd: dorm(view, sd).reset_index().rename(columns={'index':'household_key'}))
print('c2 built', tab2.shape, tab2.snapshot_day.nunique())
d2 = e3.merge(tab2.drop(columns=['snapshot_day']), on='household_key', how='left')
print('c2', d2.shape, list(d2.columns[-10:]))
api.save_table(d2, 'e012_dorm.parquet')

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api

def dorm(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    trips = w.groupby(['household_key','day']).size().reset_index()[['household_key','day']].sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key').day.diff()
    gg = trips.groupby('household_key').gap
    out = pd.DataFrame(index=hh)
    out['gap_mean84'] = gg.mean()
    out['gap_max84'] = gg.max()
    out['gap_std84'] = gg.std()
    out['gap_last'] = gg.last()
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        s = spend[(spend.day > d-28*(k+1)) & (spend.day <= d-28*k)].groupby('household_key').sales_value.sum()
        cnt = cnt + (s.reindex(hh).fillna(0) <= 0).astype(float)
    out['n_zero_l6'] = cnt
    last = trips.groupby('household_key').day.max()
    out['inactive_run84'] = np.maximum(out['gap_max84'].fillna(84), (d - last).where(last.notna(), 84))
    ten = tx.groupby('household_key').day.min()
    out['tenure'] = (d - ten).clip(lower=1)
    wk_act = tx.assign(wk=(tx.day+8)//7).groupby(['household_key','wk']).size()
    n_act_weeks = wk_act.groupby('household_key').size()
    n_weeks_ten = ((d - ten) // 7).clip(lower=1)
    out['active_week_share'] = (n_act_weeks / n_weeks_ten).clip(0,1)
    out['zero_frac_full'] = 1 - out['active_week_share']
    return out.reset_index().rename(columns={'index':'household_key'})

tab2 = api.build_features(dorm)
e3 = api.load_saved('e003_catmix.parquet')
d2 = e3.merge(tab2, on=['household_key','snapshot_day'], how='left')
print('c2', d2.shape, d2.household_key.nunique())
assert d2.household_key.duplicated(['household_key','snapshot_day']).sum()==0
api.save_table(d2, 'e012_dorm.parquet')

# --- candidate 3: peer anchors ---
def peers(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    def win(lo, hi):
        return spend[(spend.day > d-hi) & (spend.day <= d-lo)].groupby('household_key').sales_value.sum()
    r28 = win(0,28).reindex(hh).fillna(0)
    r56 = win(28,56).reindex(hh).fillna(0)
    r84 = win(56,84).reindex(hh).fillna(0)
    out = pd.DataFrame(index=hh)
    out['peer_recent28'] = r28.mean()
    out['peer_recent28_med'] = r28.median()
    out['peer_ratio'] = r28 / max(r56.mean(), 1e-9)
    out['peer_ratio2'] = r56 / max(r84.mean(), 1e-9)
    out['peer_p90'] = r28.quantile(.9)
    out['peer_p10'] = r28.quantile(.1)
    # prior-snapshot cohort mean (train snapshots are every 28d)
    prior = d - 28
    sp = spend[(spend.day > prior-28) & (spend.day <= prior)]
    out['cohort_prior4w'] = sp.groupby('household_key').sales_value.sum().mean()
    # seasonal anchor: same 4w window one year earlier (population level)
    ly = spend[(spend.day > d-363-28) & (spend.day <= d-336)]
    out['anchor_ly_pop'] = ly.sales_value.sum() if len(ly) else np.nan
    # per-household last-year same-window spend
    ly_hh = ly.groupby('household_key').sales_value.sum()
    out['spend_ly4w'] = ly_hh.reindex(hh).fillna(0)
    out['has_ly4w'] = ly_hh.reindex(hh).notna().astype(float)
    # population growth factor recent vs ly
    out['pop_ratio'] = r28.sum() / max(ly.sales_value.sum(), 1e-9) if len(ly) else np.nan
    return out.reset_index().rename(columns={'index':'household_key'})

tab3 = api.build_features(peers)
print('c3', tab3.shape)
d3 = e3.merge(tab3, on=['household_key','snapshot_day'], how='left')
print('c3 merged', d3.shape)
api.save_table(d3, 'e013_peers.parquet')
print(tab3.head())

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api

def peers(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    def win(lo, hi):
        return spend[(spend.day > d-hi) & (spend.day <= d-lo)].groupby('household_key').sales_value.sum()
    r28 = win(0,28).reindex(hh).fillna(0)
    r56 = win(28,56).reindex(hh).fillna(0)
    r84 = win(56,84).reindex(hh).fillna(0)
    out = pd.DataFrame(index=hh)
    out['peer_recent28'] = r28.mean()
    out['peer_recent28_med'] = r28.median()
    out['peer_ratio'] = r28 / max(r56.mean(), 1e-9)
    out['peer_ratio2'] = r56 / max(r84.mean(), 1e-9)
    out['peer_p90'] = r28.quantile(.9)
    out['peer_p10'] = r28.quantile(.1)
    prior = d - 28
    sp = spend[(spend.day > prior-28) & (spend.day <= prior)]
    out['cohort_prior4w'] = sp.groupby('household_key').sales_value.sum().mean()
    ly = spend[(spend.day > d-363-28) & (spend.day <= d-336)]
    out['anchor_ly_pop'] = ly.sales_value.sum() if len(ly) else np.nan
    ly_hh = ly.groupby('household_key').sales_value.sum()
    out['spend_ly4w'] = ly_hh.reindex(hh).fillna(0)
    out['has_ly4w'] = ly_hh.reindex(hh).notna().astype(float)
    out['pop_ratio'] = r28.sum() / max(ly.sales_value.sum(), 1e-9) if len(ly) else np.nan
    return out.reset_index().rename(columns={'index':'household_key'})

tab3 = api.build_features(peers)
print('c3', tab3.shape)
e3 = api.load_saved('e003_catmix.parquet')
d3 = e3.merge(tab3, on=['household_key','snapshot_day'], how='left')
print('c3 merged', d3.shape, 'dups', d3.duplicated(['household_key','snapshot_day']).sum())
api.save_table(d3, 'e013_peers.parquet')
d2 = api.load_saved('e012_dorm.parquet')
print('c2 check', d2.shape, 'dups', d2.duplicated(['household_key','snapshot_day']).sum(), d2.snapshot_day.nunique())
print(d2[['gap_mean84','n_zero_l6','inactive_run84','active_week_share']].describe().round(3))

# ---- cell ----
import numpy as np, pandas as pd, agent_api as api

def dorm(view, snapshot_day):
    tx = view.transactions
    hh = view.households
    d = snapshot_day
    w = tx[(tx.day > d-84) & (tx.day <= d)]
    trips = w.groupby(['household_key','day']).size().reset_index()[['household_key','day']].sort_values(['household_key','day'])
    trips['gap'] = trips.groupby('household_key').day.diff()
    gg = trips.groupby('household_key').gap
    out = pd.DataFrame(index=hh)
    out['gap_mean84'] = gg.mean()
    out['gap_max84'] = gg.max()
    out['gap_std84'] = gg.std()
    out['gap_last'] = gg.last()
    spend = tx.groupby(['household_key','day']).sales_value.sum().reset_index()
    cnt = pd.Series(0.0, index=hh)
    for k in range(6):
        s = spend[(spend.day > d-28*(k+1)) & (spend.day <= d-28*k)].groupby('household_key').sales_value.sum()
        cnt = cnt + (s.reindex(hh).fillna(0) <= 0).astype(float)
    out['n_zero_l6'] = cnt
    last = trips.groupby('household_key').day.max()
    out['inactive_run84'] = np.maximum(out['gap_max84'].fillna(84), (d - last).where(last.notna(), 84))
    ten = tx.groupby('household_key').day.min()
    out['tenure'] = (d - ten).clip(lower=1)
    wk_act = tx.assign(wk=(tx.day+8)//7).groupby(['household_key','wk']).size()
    n_act_weeks = wk_act.groupby('household_key').size()
    n_weeks_ten = ((d - ten) // 7).clip(lower=1)
    out['active_week_share'] = (n_act_weeks / n_weeks_ten).clip(0,1)
    out['zero_frac_full'] = 1 - out['active_week_share']
    return out.reset_index().rename(columns={'index':'household_key'})

tab2 = api.build_features(dorm)
e3 = api.load_saved('e003_catmix.parquet')
d2 = e3.merge(tab2, on=['household_key','snapshot_day'], how='left')
print('c2', d2.shape, 'snapdays', d2.snapshot_day.nunique(), 'dups', d2.duplicated(['household_key','snapshot_day']).sum())
api.save_table(d2, 'e012_dorm.parquet')
# NaN rates
for c in ['gap_mean84','gap_max84','gap_std84','gap_last','inactive_run84']:
    print(c, 'nan %.3f' % d2[c].isna().mean())