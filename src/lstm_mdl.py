from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from textwrap import wrap
from sklearn.preprocessing import MinMaxScaler
import math

class LSTM_mdl:

    def __init__(self, data):
        print("Initializing the lstm predictor")
        self.data = data

        # fix random seed for reproducibility
        np.random.seed(7)
        look_back = 3

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(10, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        # normalize the dataset
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # self.data = scaler.fit_transform(self.data)
        minm = np.min(self.data)
        maxm = np.max(self.data)
        self.data = (self.data - minm) / (maxm-minm)

        # #split dataset in train and testing set
        x = self.data[:, :-1]
        y = self.data[:, -1]
        X_train, X_test, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=0)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        print(trainPredict.shape)
        print(testPredict.shape)

        # # invert predictions
        # trainPredict = scaler.inverse_transform(trainPredict)
        # trainY = scaler.inverse_transform([trainY])
        # testPredict = scaler.inverse_transform(testPredict)
        # testY = scaler.inverse_transform([testY])
        trainPredict = trainPredict * (maxm-minm) + minm
        trainY = trainY * (maxm-minm) + minm
        testPredict = testPredict * (maxm-minm) + minm
        testY = testY * (maxm-minm) + minm

        print(trainPredict.shape)
        print(testPredict.shape)


        # calculate root mean squared error
        trainScore = math.sqrt(metrics.mean_squared_error(trainY, trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(metrics.mean_squared_error(testY, testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))

        print('Mean Absolute Error:', metrics.mean_absolute_error(testY, testPredict[:, 0]))
        print('Mean Squared Error:', metrics.mean_squared_error(testY, testPredict[:, 0]))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(testY, testPredict[:, 0])))
        print('R2 score:', metrics.r2_score(testY, testPredict[:, 0]))
        print("Correlation:", pearsonr(testY, testPredict[:, 0])[0])

        # df = pd.DataFrame({'sub_center_id': test_anm, 'Actual': test_y.flatten(), 'Predicted': y_pred.flatten()})
        # df.to_csv('test1.csv')

        fig, ax1 = plt.subplots(figsize=(4, 3))

        # plt.plot(test_y, 'go', label="True score")
        # plt.plot(y_pred, 'bo', label="Predicted score")
        inds = testY.argsort()
        sorted_y_test = testY[inds]
        sorted_y_pred = testPredict[:, 0][inds]
        plt.scatter(np.arange(len(testY)), sorted_y_test, marker='^', label="True score")
        plt.scatter(np.arange(len(testPredict[:, 0])), sorted_y_pred, marker='x', label="Predicted score")

        plt.xlabel('Anonymized HW')
        plt.ylabel('Non-diligence score')
        plt.legend(loc="upper left")
        plt.title("\n".join(wrap("True score and predicted scores", 100)))

        plt.savefig("fig/lstm_true_pred", bbox_inches="tight", dpi=100)
        # plt.show()