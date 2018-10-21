import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

print('** SVM Classification *******************************************************')

landsatData = pd.read_csv("./resources/landsat/lantsat.csv")

landsatData.describe()

X_landSatAllFeatures = landsatData.iloc[:, np.arange(36)].copy()

y_midPixelAsTarget = landsatData.iloc[:, 36].copy()

# Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
allFeaturesIndexes = X_landSatAllFeatures.index
targetData = y_midPixelAsTarget
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(allFeaturesIndexes, targetData):
    train_ind, test_ind = allFeaturesIndexes[train_index], allFeaturesIndexes[test_index]

Test_Matrix = X_landSatAllFeatures.loc[test_ind]
Test_Target_Matrix = y_midPixelAsTarget.loc[test_ind]
Train_Matrix = X_landSatAllFeatures.loc[train_ind]
Train_Target_Matrix = y_midPixelAsTarget.loc[train_ind]

scaler = StandardScaler().fit(Train_Matrix)
Train_Matrix, Test_Matrix = scaler.transform(Train_Matrix), scaler.transform(Test_Matrix)

# data training with hyperparameter tuning for C
clf = SVC()

param_grid = [
    {'kernel': ['rbf'], 'C': [2 ** x for x in range(0, 6)]},
]
inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
grid_search = GridSearchCV(clf, param_grid, cv=inner_cv, n_jobs=1, scoring='accuracy', verbose=3)
grid_search.fit(Train_Matrix, Train_Target_Matrix)

clf = grid_search.best_estimator_
# data testing
T_predict = clf.predict(Test_Matrix)



print("SVM: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(100 * accuracy_score(Test_Target_Matrix, T_predict)))
print(grid_search.best_params_)
print(grid_search.param_grid)

