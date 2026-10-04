import pandas as pd, numpy as np
tt = train_targets()
y = tt.future_spend_4w.values
gm = tt.groupby('household_key')['future_spend_4w']
hmean = gm.transform('mean'); hsum = gm.transform('sum'); hsize = gm.transform('size')
hstd = gm.transform('std')

print('mean within-hh CV:', (hstd/hmean.clip(lower=1)).mean().round(3))
for k in [0.5,1,2,4]:
    shrunk = (hsum + k*136.86) / (hsize + k)
    print('shrinkage k=%s: MAE=%.2f' % (k, np.mean(np.abs(shrunk.values-y))))

print('\nhh-mean distribution:')
print(gm.mean().describe([.1,.25,.5,.75,.9,.95]).round(1))
print('\nhh obs counts:', gm.size().describe([.1,.5,.9]).to_dict())