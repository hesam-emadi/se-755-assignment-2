import numpy as np

import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

print('** DNN Classification *******************************************************')

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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

X_train = Train_Matrix
y_train = Train_Target_Matrix
X_test = Test_Matrix
y_test = Test_Target_Matrix

# X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]

xx, yy = Train_Matrix.shape
#training phase
feature_cols = [tf.feature_column.numeric_column("X", shape=[36])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=7, feature_columns=feature_cols)
# dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300,100], n_classes=10)


input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=40, batch_size=64, shuffle=True)
dnn_clf.train(input_fn=input_fn)

#testing phase
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_test}, y=y_test, shuffle=False)
eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
print("The prediction result is : {0:.2f}%".format(100*eval_results['accuracy']))
y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
y_pred = list(y_pred_iter)
y_pred[0]


print('**********************************************************************************')