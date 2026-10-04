import agent_api, pandas as pd, numpy as np

# Prototype new features at snapshot d=431 (outside build_features, capped view is fine for prototyping)
d = 431
v = agent_api.snapshot(d)
tx = v.transactions
t = agent_api.load_saved('e001_history.parquet')
hh = t.loc[t.snapshot_day==d, 'household_key'].values
print('n hh', len(hh))

tx = tx[tx.household_key.isin(set(hh))]
S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
days = np.arange(1, d+1)
S = S.reindex(columns=days, fill_value=0.0).reindex(hh, fill_value=0.0)
M = S.values.astype(np.float64)
C = np.zeros((M.shape[0], d+1)); C[:,1:] = M.cumsum(1)
idx = {h:i for i,h in enumerate(S.index)}

def spend(a, b):
    a = max(a,1); b = min(b,d)
    if a>b: return np.zeros(M.shape[0])
    return C[:,b]-C[:,a-1]

def row(h): return idx[h]

# windows j: [d-28j, d-28j+27]
sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])  # 13 x n
tj = np.stack([spend(d-28*j-84, d-28*j-1) for j in range(1,9)])
w = 0.5**np.arange(13)[:,None]  # 13 x 1
act = (sj>0).astype(float)
p_active13 = (w*act).sum(0)/w.sum()
p_active6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
usual_active = (w*sj*act).sum(0)/np.maximum(w*act).sum(0) if False else (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
exp_dec = (w*sj).sum(0)/w.sum()
ratios = np.clip(sj[:8]/(tj+10.0), 0, 3)
persist = (0.5**np.arange(8)[:,None]*ratios).sum(0)/0.5**np.arange(8).sum()
# streak
streak = np.zeros(M.shape[0])
for j in range(13):
    streak += act[j]*(streak>=j)
# hazard: P(active j | active j-1), j=2..13
num = (act[1:]*act[:-1]).sum(0); den = act[:-1].sum(0)
hazard = num/np.maximum(den,1e-9)
# cv
m13 = sj.mean(0); sd13 = sj.std(0); cv13 = sd13/(m13+1.0)
# last-year window [d-363, d-336]
ly = spend(d-363, d-336); ly_avail = (d>=364).astype(float)*np.ones(M.shape[0])
ly_ratio = ly/(sj[0]+10.0)
# gaps within [d-364, d]: use active days
Act = (M>0)
lastd = np.array([np.max(np.where(Act[i])[0])+1 if Act[i].any() else 0 for i in range(M.shape[0])])
# longest gap in last 364 days
gap_long = np.zeros(M.shape[0]); gap_n21 = np.zeros(M.shape[0])
for i in range(M.shape[0]):
    ad = np.where(Act[i])[0]
    ad = ad[(ad>=d-364)]
    if len(ad)==0: continue
    gaps = np.diff(np.concatenate([[-1], ad, [d]])) - 1  # gaps between active days incl edges
    gap_long[i] = gaps.max(); gap_n21[i] = (gaps>=21).sum()

out = pd.DataFrame({
 'p_active13':p_active13,'p_active6':p_active6,'usual_active':usual_active,'exp_dec':exp_dec,
 'persist':persist,'streak':streak,'hazard':hazard,'cv13':cv13,'ly_spend':ly,'ly_ratio':ly_ratio,
 'gap_long':gap_long,'gap_n21':gap_n21,'expected':p_active13*usual_active}, index=S.index)
print(out.describe().round(3).T[['mean','std','min','50%','max']])
print('corr with target:')
tt = agent_api.train_targets()
mg = out.reset_index().rename(columns={'index':'household_key'})
mg['snapshot_day']=d
m2 = mg.merge(tt[tt.snapshot_day==d], on=['household_key','snapshot_day'])
print(m2.drop(columns=['household_key','snapshot_day']).corr()['future_spend_4w'].round(3))