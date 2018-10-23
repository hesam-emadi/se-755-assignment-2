#
# from sklearn.cluster import KMeans
# import numpy as np
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# print kmeans.labels_
# # array([0, 0, 0, 1, 1, 1], dtype=int32)
# print kmeans.predict([[0, 0], [4, 4]])
# print kmeans.labels_
# # array([0, 1], dtype=int32)
# print kmeans.cluster_centers_
# # array([[1., 2.],[4., 2.]])
#
# #===================
#
# import tensorflow as tf
# x = tf.Variable(3, name="x")
# y = tf.Variable(4, name="y")
# f = x*x*y + y + 2
#
# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result =  sess.run(f)
# print result

import numpy as np
import tensorflow as tf
# To plot pretty figures
import matplotlib.pyplot as plt
from datetime import datetime
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

#retrieving data (the first step always)
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]