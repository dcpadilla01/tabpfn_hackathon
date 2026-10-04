import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
print(e3.groupby('snapshot_day').tenure.describe()[['min','25%','50%','75%','max']])

# how was spend_l13 filled when tenure short?
sub = e3[e3.tenure < 364]
print('tenure<364 rows:', len(sub), 'spend_l13 stats:', sub.spend_l13.describe()[['min','mean','max']].to_dict())
print('spend_l13 == 0 share among tenure<364:', float((sub.spend_l13==0).mean()))
print('spend_l13 == spend_l1 share:', float(np.isclose(sub.spend_l13, sub.spend_l1).mean()))
print('spend_l13 == spend_l123_mean share:', float(np.isclose(sub.spend_l13, sub.spend_l123_mean).mean()))
# among tenure>=364, is l13 nonzero?
sub2 = e3[e3.tenure >= 364]
print('tenure>=364 rows:', len(sub2), 'l13 zero share:', float((sub2.spend_l13==0).mean()))
print(e3.groupby('snapshot_day').apply(lambda g: pd.Series({'n':len(g), 'tenure>=364': float((g.tenure>=364).mean())}), include_groups=False))
