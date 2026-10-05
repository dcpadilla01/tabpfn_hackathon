
import pandas as pd, numpy as np, agent_api as A

tt = train_targets()
f = load_saved('feats_v4.parquet')
p13 = load_saved('pred_e013.parquet')
print('p13 dtypes:', p13.dtypes.to_dict())
print('f dtypes:', f.dtypes.head(3).to_dict(), '| tt dtypes:', tt.dtypes.to_dict())
print('p13 head:'); print(p13.head(3))

p13 = p13.rename(columns={'prediction':'p13'})
m = f.merge(p13, on=['household_key','snapshot_day'], how='inner')
print('merged feats+pred:', len(m))
tr = m.merge(tt, on=['household_key','snapshot_day'])
print('merged with targets:', len(tr))
if len(tr):
    tr['err'] = (tr.future_spend_4w - tr.p13).abs()
    tr['dec'] = pd.qcut(tr.p13, 10, duplicates='drop')
    g = tr.groupby('dec', observed=True).agg(n=('err','size'), pred=('p13','mean'), y=('future_spend_4w','mean'),
                                             ymed=('future_spend_4w','median'), mae=('err','mean'))
    print(g.round(1).to_string())
    print('overall train MAE %.2f' % tr.err.mean())
    for z in [0,1]:
        s = tr[tr.future_spend_4w==0] if z==0 else tr[tr.future_spend_4w>0]
        print('actual %s: n=%d MAE %.2f meanpred %.1f' % ('zero' if z==0 else '>0', len(s), s.err.mean(), s.p13.mean()))
