from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
import time
import utils
import pdb

krkopt_data = utils.load_data()
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1)

################# Phase 1: Simple base learner ###########################
base = DecisionTreeClassifier(min_samples_split=5, max_depth=15)

# Setup 1: Base only
start_time = time.time()
scores = cross_val_score(base, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('DecisionTree: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 2: Boost
start_time = time.time()
model_2 = AdaBoostClassifier(base_estimator=base, n_estimators=100)
scores = cross_val_score(model_2, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('AdaBoost: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 3: Bagging
start_time = time.time()
model_3 = BaggingClassifier(base_estimator=base, n_estimators=100)
scores = cross_val_score(model_3, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Bagging: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 4: Multiboost
start_time = time.time()
sub_committee = AdaBoostClassifier(base_estimator=base, n_estimators=10)
model_4 = BaggingClassifier(base_estimator=sub_committee, n_estimators=10)
scores = cross_val_score(model_4, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Multiboost: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 5: Iterative Bagging
start_time = time.time()
sub_committee = BaggingClassifier(base_estimator=base, n_estimators=10)
model_5 = AdaBoostClassifier(base_estimator=sub_committee, n_estimators=10)
scores = cross_val_score(model_5, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Iterative Bagging: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

################## Phase 2: Complex base learner ###########################
base = DecisionTreeClassifier(min_samples_split=2, max_depth=20)

# Setup 1: Base only
start_time = time.time()
scores = cross_val_score(base, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('DecisionTree: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 2: Boost
start_time = time.time()
model_2 = AdaBoostClassifier(base_estimator=base, n_estimators=100)
scores = cross_val_score(model_2, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('AdaBoost: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 3: Bagging
start_time = time.time()
model_3 = BaggingClassifier(base_estimator=base, n_estimators=100)
scores = cross_val_score(model_3, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Bagging: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 4: Multiboost
start_time = time.time()
sub_committee = AdaBoostClassifier(base_estimator=base, n_estimators=10)
model_4 = BaggingClassifier(base_estimator=sub_committee, n_estimators=10)
scores = cross_val_score(model_4, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Multiboost: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))

# Setup 5: Iterative Bagging
start_time = time.time()
sub_committee = BaggingClassifier(base_estimator=base, n_estimators=10)
model_5 = AdaBoostClassifier(base_estimator=sub_committee, n_estimators=10)
scores = cross_val_score(model_5, krkopt_data['data'], krkopt_data['target'], cv=sss)
print('Iterative Bagging: {}'.format(scores.mean()))
print('Time Elapsed: {}\n'.format(time.time() - start_time))


################ Phase 1 Result #######################

# DecisionTree: 0.7693870277975766
# Time Elapsed: 0.3435828685760498

# AdaBoost: 0.9261582323592302
# Time Elapsed: 49.13990235328674

# Bagging: 0.8382751247327157
# Time Elapsed: 20.88724660873413

# Multiboost: 0.9019957234497505
# Time Elapsed: 39.02409052848816

# Iterative Bagging: 0.8755167498218104
# Time Elapsed: 27.958540439605713


################ Phase 2 Result #######################

# DecisionTree: 0.8565217391304347
# Time Elapsed: 0.37897205352783203

# AdaBoost: 0.8598004276550248
# Time Elapsed: 5.261999130249023

# Bagging: 0.8817533856022809
# Time Elapsed: 23.280755519866943

# Multiboost: 0.8863863150392015
# Time Elapsed: 20.83625864982605

# Iterative Bagging: 0.8923378474697078
# Time Elapsed: 30.87542700767517
