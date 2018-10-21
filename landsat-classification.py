import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

print('** LogisticRegression Classification *******************************************************')

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


classifier = LogisticRegression()

# hyperparameter tuning for solver and C
param_grid = [
    {'solver': ['newton-cg'], 'C': [1, 2, 4, 8, 16, 32]},
    {'solver': ['lbfgs'], 'C': [1, 2, 4, 8, 16, 32]},
    {'solver': ['liblinear'], 'C': [1, 2, 4, 8, 16, 32]},
    {'solver': ['sag'], 'C': [1, 2, 4, 8, 16, 32]},
    {'solver': ['saga'], 'C': [1, 2, 4, 8, 16, 32]},
]

inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, n_jobs=1, scoring='accuracy', verbose=3)
grid_search.fit(Train_Matrix, Train_Target_Matrix)

classifier = grid_search.best_estimator_
# data testing
T_predict = classifier.predict(Test_Matrix)

print('** LogisticRegression Result *******************************************************')

print("Logistic Regression: The prediction accuracy (tuned) for all testing sentence is : {:.2f}%.".format(100 * accuracy_score(Test_Target_Matrix, T_predict)))
print(grid_search.best_params_)
print(grid_search.param_grid)

print('**********************************************************************************')