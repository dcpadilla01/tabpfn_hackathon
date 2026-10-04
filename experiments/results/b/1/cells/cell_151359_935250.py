import agent_api, pandas as pd, numpy as np

# Hypothesis E007: future 4w spend = P(active in next 4w) x usual spend when active.
# Explicit decomposition: activity probability over trailing 28d windows, usual per-active-window
# spend, streak, hazard, gaps, last-year same window.

def compute_new(view, d, hh_keys):
    tx = view.transactions
    tx = tx[tx.household_key.isin(set(hh_keys))]
    S = tx.groupby(['household_key','day']).sales_value.sum().unstack(fill_value=0.0)
    S = S.reindex(columns=np.arange(1,d+1), fill_value=0.0).reindex(hh_keys, fill_value=0.0)
    M = S.values.astype(np.float64)
    n = M.shape[0]
    C = np.zeros((n, d+1)); C[:,1:] = M.cumsum(1)
    def spend(a,b):
        a=max(a,1); b=min(b,d)
        if a>b: return np.zeros(n)
        return C[:,b]-C[:,a-1]
    sj = np.stack([spend(d-28*j, d-28*j+27) for j in range(1,14)])
    act = (sj>0).astype(float)
    w = 0.5**np.arange(13)[:,None]
    p13 = (w*act).sum(0)/w.sum()
    p6 = (w[:6]*act[:6]).sum(0)/w[:6].sum()
    p3 = act[:3].mean(0)
    usual = (w*sj*act).sum(0)/np.maximum((w*act).sum(0),1e-9)
    expected = p13*usual
    streak = np.zeros(n)
    for j in range(13): streak += act*j*(streak>=j)  # placeholder fixed below
    streak = np.zeros(n)
    for j in range(13): streak += act[j]*(streak>=j)
    hazard = (act[1:]*act[:-1]).sum(0)/np.maximum(act[:-1].sum(0),1e-9)
    cv13 = sj.std(0)/(sj.mean(0)+1.0)
    Act = M>0
    gap_long = np.zeros(n); gap_n21 = np.zeros(n)
    for i in range(n):
        ad = np.where(Act[i])[0]; ad = ad[ad>=d-364]
        if len(ad)==0: continue
        gaps = np.diff(np.concatenate([[-1], ad, [d]]))-1
        gap_long[i]=gaps.max(); gap_n21[i]=(gaps>=21).sum()
    ly = spend(d-363, d-336) if d>=364 else np.zeros(n)
    ly_ratio = ly/(sj[0]+10.0)
    rvu = sj[0]/(usual+10.0)
    out = pd.DataFrame({
        'p_active13':p13,'p_active6':p6,'p_act3':p3,'usual_active':usual,'expected':expected,
        'log_expected':np.log1p(expected),'streak':streak,'hazard':hazard,'cv13':cv13,
        'gap_long':gap_long,'gap_n21':gap_n21,'ly_spend':ly,'ly_ratio':ly_ratio,'recent_vs_usual':rvu},
        index=pd.Index(hh_keys, name='household_key'))
    return out

def fn(view, snapshot_day):
    d = snapshot_day
    hh = view.households
    if hh is None:
        tx = view.transactions
        fd = tx.groupby('household_key').day.min()
        hh = fd[fd <= d-84].index.values
    else:
        hh = hh.household_key.values if hasattr(hh,'household_key') else np.asarray(hh)
    return compute_new(view, d, hh)

F = agent_api.build_features(fn)
print('build ok', F.shape)
print(F.columns.tolist())
e1 = agent_api.load_saved('e001_history.parquet')
for d in [95, 207, 431, 543]:
    print(d, len(F[F.snapshot_day==d]), len(e1[e1.snapshot_day==d]))
agent_api.save_table(F, 'e007_new.parquet')