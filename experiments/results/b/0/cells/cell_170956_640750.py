import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
mac = agent_api.load_saved('macro.parquet')
if 'household_key' not in mac.columns: mac = mac.reset_index()
print('macro index/cols:', mac.columns.tolist()[:4], '... nunique days:', mac.snapshot_day.nunique())
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

eval_table(df, 'e018 baseline')
dm = df.merge(mac[[c for c in mac.columns if c not in df.columns] + ['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='left')
eval_table(dm, 'e018 + macro day-level cols')

# zero-target prototype: recency-based zero-risk gating
d0 = df.copy()
d0['zero_risk'] = np.clip((d0.recency-7)/21, 0, 1)
eval_table(d0.assign(zr_pred=d0.spend28*0), 'sanity')  # noop
d1 = df.copy()
d1['zero_risk'] = np.clip((d1.recency-7)/21, 0, 1)
eval_table(d1, 'e018 + zero_risk(recency)')