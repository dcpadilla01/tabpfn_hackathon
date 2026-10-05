import agent_api, pandas as pd, numpy as np, xgboost as xgb, warnings, time
warnings.filterwarnings('ignore')
f3 = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = f3.merge(tt, on=['household_key','snapshot_day'], how='inner')
drop = ['household_key','snapshot_day','future_spend_4w']
feats = [c for c in df.columns if c not in drop]
days = sorted(df.snapshot_day.unique())

def mk(obj='reg:absoluteerror',n=1200,lr=0.03,md=7,mcw=10,ss=0.8,cs=0.7,rl=1.0):
    return xgb.XGBRegressor(n_estimators=n,learning_rate=lr,max_depth=md,min_child_weight=mcw,
        subsample=ss,colsample_bytree=cs,reg_lambda=rl,objective=obj,tree_method='hist',n_jobs=8,random_state=0)

def oof_preds(obj='reg:absoluteerror',**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(obj=obj,**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

# quick: does adding day index as feature help? and does dropping weak features help?
t0=time.time()
df['day_idx'] = df.snapshot_day.astype(float)
feats2 = feats+['day_idx']
pD,m = oof_preds(n=1200,lr=0.03,md=7,mcw=10)  # base cfg but mcw10 (new best-ish)
# rebuild with feats2
def oof_preds2(fl,**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**kw)
        m.fit(tr[fl],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[fl]); seen[df.index.isin(va.index)]=1
    return oof, seen>0
pE,mE = oof_preds2(feats2,n=1200,lr=0.03,md=7,mcw=10)
print('base mcw10', round(mae(pD,m),3))
print('with day_idx', round(mae(pE,mE),3))
