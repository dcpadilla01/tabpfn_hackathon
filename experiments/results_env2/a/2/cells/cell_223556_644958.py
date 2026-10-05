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

def oof_preds(**kw):
    oof = np.zeros(len(df)); seen=np.zeros(len(df))
    for d in days:
        va = df[df.snapshot_day==d]; tr = df[df.snapshot_day<d]
        if len(tr)==0: continue
        m = mk(**kw)
        m.fit(tr[feats],tr.future_spend_4w,verbose=False)
        oof[df.index.isin(va.index)]=m.predict(va[feats]); seen[df.index.isin(va.index)]=1
    return oof, seen>0

def mae(p,mask):
    y=df.future_spend_4w.values[mask]; return np.abs(p[mask]-y).mean()

t0=time.time()
# final: average of md7 lr.02 n2000 and md6 base — the two best configs — full-train models for val days
preds_val = []
for kw in [dict(n=2000,lr=0.02,md=7,mcw=10), dict(n=1200,lr=0.03,md=6,mcw=10)]:
    m = mk(**kw)
    tr = df[df.snapshot_day<=431]
    m.fit(tr[feats],tr.future_spend_4w,verbose=False)
    fv = agent_api.load_saved('feats_v3.parquet')
    fv = fv[fv.snapshot_day>=459]
    preds_val.append(m.predict(fv[feats]))
p = np.mean(preds_val,axis=0)
out = fv[['household_key','snapshot_day']].copy()
out['prediction']=p
path = agent_api.save_table(out,'pred_e004.parquet')
print(path, out.shape, out.prediction.describe().round(2))
