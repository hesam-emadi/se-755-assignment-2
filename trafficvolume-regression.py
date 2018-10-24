import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import explained_variance_score, mean_absolute_error, median_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt

def conductBayesianLinearRegression(trainingData, testData, trainingTarget, testTarget):
    # Create Bayesian linear regression object
    regr = linear_model.BayesianRidge()

    # Train the model using the training sets
    regr.fit(trainingData, trainingTarget)

    # Make predictions using the testing set
    testDataPrediction = regr.predict(testData)
    # Make predictions using the testing set
    trainingDataPrediction = regr.predict(trainingData)

    # Trying to calculate precision and recall
    # score = regr.decision_function(testData)
    #     print("average_precision_score: %.4f"
    #           % average_precision_score(testTarget, score))


    lr = linear_model.BayesianRidge().fit(trainingData,trainingTarget)
    testDataPrediction=lr.predict(testData)
    print(lr.score(testData, testTarget))
    # print model
    plt.scatter(range(len(testData)), testTarget,  color='black')
    # Prediction and draw the diagram
    plt.plot(range(len(testData)), testDataPrediction, color='red', linewidth=1)
    plt.legend(["predict", "true"], loc='upper right')
    plt.title('Bayesian Linear Regression')
    plt.show()

    print(' ')
    # The coefficients
    print('Coefficients and Intercept are: ', regr.coef_, "   ", regr.intercept_, ' respectively')
    # The mean squared error
    print('** Bayesian Linear Regression *******************************************************')
    print("Mean squared error for testing data: %.4f" % mean_squared_error(testTarget, testDataPrediction))

    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.4f' % r2_score(testTarget, testDataPrediction))
    print('Explained variance regression score function: %.4f' % explained_variance_score(testTarget, testDataPrediction))
    print('Mean absolute error regression loss: %.4f' % mean_absolute_error(testTarget, testDataPrediction))
    print('Median absolute error regression loss: %.4f' % median_absolute_error(testTarget, testDataPrediction))

    print("Mean squared error for training data: %.4f" % mean_squared_error(trainingTarget, trainingDataPrediction))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for training data: %.4f' % r2_score(trainingTarget, trainingDataPrediction))
    print('Explained variance regression score function: %.4f' % explained_variance_score(trainingTarget, trainingDataPrediction))
    print('Mean absolute error regression loss: %.4f' % mean_absolute_error(trainingTarget, trainingDataPrediction))
    print('Median absolute error regression loss: %.4f' % median_absolute_error(trainingTarget, trainingDataPrediction))

    print('******************************************************* ')
    classifier = linear_model.BayesianRidge()
    param_grid = [
        {'alpha_1': [1.e-4, 1.e-6, 1.e-8], 'alpha_2': [1.e-4, 1.e-6, 1.e-8],
         'lambda_1': [1.e-4, 1.e-6, 1.e-8], 'lambda_2': [1.e-4, 1.e-6, 1.e-8]}
    ]
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=1)
    grid_search = GridSearchCV(classifier, param_grid, cv=inner_cv, scoring='neg_mean_squared_error', verbose=3)

    grid_search.fit(trainingData, trainingTarget)

    classifier = grid_search.best_estimator_

    y_trainingDataPrediction_tuned = classifier.predict(trainingData)
    y_testDataPrediction_tuned = classifier.predict(testData)

    print(' ')
    # The coefficients
    print('Coefficients and Intercept are: ', classifier.coef_, "   ", classifier.intercept_, ' respectively')
    # The mean squared error
    print('** Tuned Bayesian Linear Regression *******************************************************')
    print('** Best params: ')
    print(grid_search.best_params_)
    print(grid_search.param_grid)

    print("Mean squared error for testing data: %.4f" % mean_squared_error(testTarget, y_testDataPrediction_tuned))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for testing data: %.4f' % r2_score(testTarget, y_testDataPrediction_tuned))
    print('Explained variance regression score function: %.4f' % explained_variance_score(testTarget, y_testDataPrediction_tuned))
    print('Mean absolute error regression loss: %.4f' % mean_absolute_error(testTarget, y_testDataPrediction_tuned))
    print('Median absolute error regression loss: %.4f' % median_absolute_error(testTarget, y_testDataPrediction_tuned))
        
    print("Mean squared error for training data: %.4f" % mean_squared_error(trainingTarget, y_trainingDataPrediction_tuned))
    # Explained variance score: 1 is perfect prediction
    print('Variance score for training data: %.4f' % r2_score(trainingTarget, y_trainingDataPrediction_tuned))
    print('Explained variance regression score function: %.4f' % explained_variance_score(trainingTarget, y_trainingDataPrediction_tuned))
    print('Mean absolute error regression loss: %.4f' % mean_absolute_error(trainingTarget, y_trainingDataPrediction_tuned))
    print('Median absolute error regression loss: %.4f' % median_absolute_error(trainingTarget, y_trainingDataPrediction_tuned))


traffic = pd.read_csv("./resources/traffic-flow/traffic_flow_data.csv")
#
# Separating feature from target
trafficAllFeatures = traffic.iloc[:, np.arange(450)].copy()

# target which is section 23 at t+1 time
targetData = traffic.iloc[:, 450].copy()

# Split the data into training/testing sets
trafficFeatureTrainingData, trafficTestData, trafficTargetTrainingData, trafficTestTarget = \
    train_test_split(trafficAllFeatures, targetData, test_size=0.2, random_state=1)

conductBayesianLinearRegression(trafficFeatureTrainingData, trafficTestData, trafficTargetTrainingData, trafficTestTarget)
