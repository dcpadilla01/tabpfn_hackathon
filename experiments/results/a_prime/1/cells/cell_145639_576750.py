import pandas as pd, numpy as np, agent_api as A

tr_days = A.snapshot_days()['train']
b = A.baseline_features()
tt = A.train_targets()
bfeats = [c for c in b.columns if c not in ('household_key','snapshot_day')]
print('E000 feats:', bfeats)
db = b.merge(tt, on=['household_key','snapshot_day'])
X = db[bfeats]
# dummies for categorical
Xn = pd.get_dummies(X.astype(object).where(~X.isna(), X), dummy_na=True)
Xn = Xn.astype(float)
y = db.future_spend_4w.values; days = db.snapshot_day.values
mu = Xn.mean(); sd = Xn.std(); sd[sd==0]=1
Xs = ((Xn-mu)/sd).fillna(0).values
errs=[]
for d in tr_days:
    m = days!=d
    A_ = np.hstack([np.ones((m.sum(),1)), Xs[m]])
    R = A_.T@A_ + 30*np.eye(A_.shape[1]); R[0,0]-=30
    w = np.linalg.solve(R, A_.T@y[m])
    Xo = np.hstack([np.ones(((~m).sum(),1)), Xs[~m]])
    errs.append(np.mean(np.abs(Xo@w - y[~m])))
print('E000 ridge LOSO MAE:', round(float(np.mean(errs)),2), '(harness E000: 92.446)')

# conditional median structure
t3 = A.load_saved('e003_catmix.parquet')
df = t3.merge(tt, on=['household_key','snapshot_day'])
df['naive'] = df[['spend_l1','spend_l2','spend_l3']].mean(1)
tr = df[df.snapshot_day.isin(tr_days)]
va = df[~df.snapshot_day.isin(tr_days)]
print('\nval rows:', len(va), 'val snap days:', sorted(va.snapshot_day.unique()))

def pava(score, y, w=None):
    # isotonic regression (weighted, increasing)
    n=len(score); w=np.ones(n) if w is None else w
    o=np.argsort(score, kind='stable'); xs=score[o]; ys=y[o].astype(float); ws=w[o]
    # pool adjacent violators
    vy=[]; vw=[]; 
    for i in range(n):
        vy.append(ys[i]); vw.append(ws[i])
        while len(vy)>1 and vy[-2]>vy[-1]:
            v2=(vy[-2]*vw[-2]+vy[-1]*vw[-1]); w2=vw[-2]+vw[-1]
            vy[-2:]=[]; vw[-2:]=[]
            vy.append(v2); vw.append(w2)
    out=np.empty(n)
    idx=0
    # expand blocks
    blocks=[]
    cnt=0
    for v,ww in zip(vy,vw):
        k=int(round(ww)) if np.allclose(ws,1) else None
        blocks.append((v,ww))
    # reconstruct via cumulative weights
    res=np.empty(n); pos=0
    for v,ww in zip(vy,vw):
        k=int(np.ceil(ww-1e-9))
        res[pos:pos+k]=v; pos+=k
    out[o]=res
    return out

def eval_pred(pred, name):
    print(f'{name}: train MAE {np.mean(np.abs(pred-y[df.index.isin(tr.index)])):.2f}' if False else f'{name}: val MAE {np.mean(np.abs(pred - va.future_spend_4w.values)):.2f}')

# Variant 1: isotonic on naive (fit train, apply val)
sc_tr = tr.naive.values; ytr=tr.future_spend_4w.values
fit = pava(np.sort(sc_tr), ytr[np.argsort(sc_tr, kind='stable')])  # placeholder
iso = pava(sc_tr, ytr)
# build step function from (sorted unique score, iso values)
o=np.argsort(sc_tr, kind='stable'); ss=sc_tr[o]; vv=iso[o]
# compress to knots
knot_x=[]; knot_y=[]
i=0
while i < len(ss):
    j=i
    while j+1<len(ss) and ss[j+1]==ss[i]: j+=1
    knot_x.append(ss[i]); knot_y.append(vv[j]); i=j+1
knot_x=np.array(knot_x); knot_y=np.array(knot_y)
def iso_apply(s):
    return np.interp(s, knot_x, knot_y)
pred_tr = iso_apply(tr.naive.values); pred_va = iso_apply(va.naive.values)
print('V1 isotonic(naive): train MAE', round(float(np.mean(np.abs(pred_tr-ytr))),2),
      'val MAE', round(float(np.mean(np.abs(pred_va-va.future_spend_4w.values))),2))
print('naive raw val MAE:', round(float(np.mean(np.abs(va.naive.values-va.future_spend_4w.values))),2))