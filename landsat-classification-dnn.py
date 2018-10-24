import numpy as np

import pandas as pd

import tensorflow as tf

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

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

X_train = Train_Matrix.values
y_train = Train_Target_Matrix.values
X_test = Test_Matrix.values
y_test = Test_Target_Matrix.values


xx, yy = Train_Matrix.shape

#training phase
feature_cols = [tf.feature_column.numeric_column("X", shape=[36])]
dnn_clf = tf.estimator.DNNClassifier(hidden_units=[256,32], n_classes=8, feature_columns=feature_cols)


input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"X": X_train}, y=y_train, num_epochs=500, batch_size=64, shuffle=True)
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