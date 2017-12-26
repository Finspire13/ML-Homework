from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import time
import utils
import pdb

krkopt_data = utils.load_data()
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)

# Setup 1: Random forest with different features
for i in range(1, 7):
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=100, max_features=i)
    scores = cross_val_score(clf, krkopt_data['data'], krkopt_data['target'], cv=sss)
    print('Random Forest {}: {}'.format(i, scores.mean()))
    print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 2: Bagging
start_time = time.time()
clf = BaggingClassifier(n_estimators=100)
scores = cross_val_score(clf, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Bagging {}: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

#################### Result #####################

# Random Forest 1: 0.8440841054882394
# Time Elapsed: 13.9739351272583

# Random Forest 2: 0.842872416250891
# Time Elapsed: 14.634706497192383

# Random Forest 3: 0.8503207412687098
# Time Elapsed: 16.625223875045776

# Random Forest 4: 0.8725944404846757
# Time Elapsed: 18.701739072799683

# Random Forest 5: 0.8784034212401997
# Time Elapsed: 20.82867693901062

# Random Forest 6: 0.885816108339273
# Time Elapsed: 23.060242652893066

# Bagging: 0.885958660014255
# Time Elapsed: 22.946104764938354
