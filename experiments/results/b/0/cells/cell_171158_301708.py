import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet').reset_index()
mday = mac.groupby('snapshot_day').first()
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))

d = df.copy()
d['m28'] = d.snapshot_day.map(mday.macro_spend28)/1e5
d['m28_p'] = d.snapshot_day.map(mday.macro_spend_p28)/1e5
d['m112'] = d.snapshot_day.map(mday.macro_spend112)/1e5
d['m_growth'] = d.snapshot_day.map(mday.macro_growth)
d['m_wk3'] = d.snapshot_day.map(mday.macro_wk_ratio_last3)
d['m_hh28'] = d.snapshot_day.map(mday.macro_hh28)/1e3
d['m_per_hh28'] = d.snapshot_day.map(mday.macro_spend_per_hh28)
eval_table(d, 'e018 + scaled macro (7)')

d2 = df.copy()
d2['m_growth'] = d2.snapshot_day.map(mday.macro_growth)
d2['m_wk3'] = d2.snapshot_day.map(mday.macro_wk_ratio_last3)
d2['m_per_hh28'] = d2.snapshot_day.map(mday.macro_spend_per_hh28)
eval_table(d2, 'e018 + macro ratios only (3)')

d3 = d.copy()
wk = (d3.snapshot_day+14+8)//7
d3['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18); d3['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
eval_table(d3, 'e018 + scaled macro + fut-week sin/cos')