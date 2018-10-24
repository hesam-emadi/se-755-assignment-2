import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn import metrics

from time import time

occupancy = pd.read_csv("./resources/occupancy-sensor/occupancy_sensor_data.csv")

occupancy.drop(['date', 'HumidityRatio'], axis=1, inplace=True)
occupancy.describe()

# occupancy attributes
X_occupancyAllFeatures = occupancy.iloc[:, np.arange(4)].copy()

# occupancy result
y_occupancyAsTarget = occupancy.iloc[:, 4].copy()

feature_prepared = X_occupancyAllFeatures

# Split the data into training/testing sets
# worldcupFeatureTrainingData, testData, worldcupTargetTrainingData, testTarget = \
#     train_test_split(feature_prepared, y_occupancyAsTarget, test_size=0.2, random_state=1)

# Testing and training sentences splitting (stratified + shuffled) based on the index (sentence ID)
allFeatures = feature_prepared.index
targetData = y_occupancyAsTarget
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

for train_index, test_index in sss.split(allFeatures, targetData):
    train_ind, test_ind = allFeatures[train_index], allFeatures[test_index]
Test_Matrix = feature_prepared.loc[test_ind]
Test_Target_Matrix = y_occupancyAsTarget.loc[test_ind]
Train_Matrix = feature_prepared.loc[train_ind]
Train_Target_Matrix = y_occupancyAsTarget.loc[train_ind]

kmeans = KMeans(init='k-means++', n_clusters=2)
t0 = time()
kmeans.fit(Train_Matrix)

kmeans.predict(Test_Matrix)

print ("Cluster Centers: ", kmeans.cluster_centers_)

print(82 * '*')
print('init\t\ttime\tinertia\t\thomo\tcompl\tv-meas\tARI\tAMI')

print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % ('k-means++', (time() - t0), kmeans.inertia_,
             metrics.homogeneity_score(Train_Target_Matrix, kmeans.labels_),
             metrics.completeness_score(Train_Target_Matrix, kmeans.labels_),
             metrics.v_measure_score(Train_Target_Matrix, kmeans.labels_),
             metrics.adjusted_rand_score(Train_Target_Matrix, kmeans.labels_),
             metrics.adjusted_mutual_info_score(Train_Target_Matrix,  kmeans.labels_)))

print(82 * '*')
