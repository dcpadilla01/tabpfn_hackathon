import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
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

# 1) recency-interaction (household-level, no leakage)
d1 = df.copy()
d1['rec_g'] = np.clip((d1.recency-7)/21, 0, 1)
for c in ['spend28','spend7','spend56','spend112','dec_spend14','rwspend84','lag_spend_1','lag_mean_1_4','spend_yoy28','lt_spend']:
    d1[c+'_g'] = d1[c]*d1.rec_g
eval_table(d1, 'e018 + recency-gated spend interactions')

# 2) recency-based zero-risk gating (raw feature)
d2 = df.copy()
d2['zero_risk'] = np.clip((d2.recency-7)/21, 0, 1)
d2['zero_risk2'] = np.clip((d2.recency-14)/28, 0, 1)
d2['zero_risk3'] = np.clip((d2.recency-7)/14, 0, 1)
eval_table(d2, 'e018 + zero_risk raw (3 variants)')

# 3) log-transform heavy-tailed spend features
sp_cols = [c for c in df.columns if c.startswith('spend') or c.startswith('lag_spend') or c.startswith('dec_') or c in ('lt_spend','rwspend84','own_ly_spend4w')]
d3 = df.copy()
for c in sp_cols:
    d3[c+'_lg'] = np.log1p(d3[c].clip(lower=0))
eval_table(d3, 'e018 + log1p(spend cols)')