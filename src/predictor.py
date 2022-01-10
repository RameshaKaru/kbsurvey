import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from textwrap import wrap
from sklearn.preprocessing import MinMaxScaler
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# from keras.models import Sequential

class Predictor:

    def __init__(self, cnfg):
        print("Initializing the predictor")
        self.cnfg = cnfg
        self.df_scores = pd.read_csv("rules/scores.csv")
        self.data = self.calc_datapoints()

    def calc_datapoints(self):
        gdf = self.df_scores.groupby("health_worker_id").agg(
            num_data=pd.NamedAgg(column='scores', aggfunc=lambda x: len(list(x))),
            scores_list=pd.NamedAgg(column='scores', aggfunc=lambda x: list(x))
        )

        print(gdf['num_data'].value_counts())

        data_points_threshold = self.cnfg['data_points_threshold']
        df = gdf[gdf['num_data'] >= data_points_threshold]
        print(df['num_data'].value_counts())

        scores_list = list(df['scores_list'])
        print(len(scores_list))
        print(len(scores_list[0]))

        data = []
        for i in range(len(scores_list)):
            d = scores_list[i]
            l = len(d)
            if l == 4:
                data.append(d)
            elif l > 4:
                for j in range(l - data_points_threshold + 1):
                    data.append(d[j:j + data_points_threshold])

        print("Number of data points", len(data))

        return np.array(data)

    def simple_predictor(self):

        # #split dataset in train and testing set
        x = self.data[:,:-1]
        y = self.data[:,-1]
        X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

        print("Total", len(X_train) + len(X_test))
        print("Train", X_train.shape)
        print("Test", X_test.shape)

        model = LinearRegression()
        model.fit(X_train, Y_train)

        print('coefficient of determination:', model.score(X_train, Y_train))

        y_pred = model.predict(X_test)
        print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))
        print('R2 score:', metrics.r2_score(Y_test, y_pred))
        print("Correlation:", pearsonr(Y_test, y_pred)[0])

        # df = pd.DataFrame({'sub_center_id': test_anm, 'Actual': test_y.flatten(), 'Predicted': y_pred.flatten()})
        # df.to_csv('test1.csv')

        fig, ax1 = plt.subplots(figsize=(4, 3))

        # plt.plot(test_y, 'go', label="True score")
        # plt.plot(y_pred, 'bo', label="Predicted score")
        inds = Y_test.argsort()
        sorted_y_test = Y_test[inds]
        sorted_y_pred = y_pred[inds]
        plt.scatter(np.arange(len(Y_test)), sorted_y_test, marker='^', label="True score")
        plt.scatter(np.arange(len(y_pred)), sorted_y_pred, marker='x', label="Predicted score")

        plt.xlabel('Anonymized HW')
        plt.ylabel('Non-diligence score')
        plt.legend(loc="upper left")
        plt.title("\n".join(wrap("True score and predicted scores", 100)))

        plt.savefig("fig/true_pred", bbox_inches="tight", dpi=100)
        # plt.show()

        return model

    # def lstmModel(lstm_units=4, num_months=3, dense_param=[2]):
    #     """
    #     Model with cmeans labels
    #     Parameters
    #     ----------
    #     num_rules : int
    #         number of rules/ number of non-diligence probabilities per ANM per time frame
    #     lstm_units : int
    #         number of lstm units
    #     num_months :  int
    #         number of months in non-diligence vector history taken as input
    #     dense_param : int
    #         dense layer parameter
    #     Returns
    #     -------
    #     model : compiled model
    #     """
    #
    #     # first_input = Input(shape=(num_months, num_rules))
    #     # second_lstm = LSTM(lstm_units)(first_input)
    #     # third_dense = Dense(dense_param[0], activation="relu")(second_lstm)
    #     # fourth_softmax = Softmax()(third_dense)
    #     #
    #     # model = Model(inputs=first_input, outputs=fourth_softmax)
    #     #
    #     # model.compile(optimizer='adam', loss='mse',
    #     #               metrics=['mse', 'mae'])
    #     #
    #     # return model
    #
    #     mdl = Sequential()
    #     mdl.add(LSTM(lstm_units, input_shape=(1, 3)))
    #     mdl.add(Dense(1))
    #     mdl.compile(loss='mean_squared_error', optimizer='adam')
    #
    #     return mdl
    #
    # def lstm_fit_pred(self, model):
    #
    #     # normalize the dataset
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     self.data = scaler.fit_transform(self.data)
    #
    #     # #split dataset in train and testing set
    #     x = self.data[:, :-1]
    #     y = self.data[:, -1]
    #     X_train, X_test, trainY, testY = train_test_split(x, y, test_size=0.2, random_state=0)
    #
    #     # reshape input to be [samples, time steps, features]
    #     trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    #     testX = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    #
    #     model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    #
    #     # make predictions
    #     trainPredict = model.predict(trainX)
    #     testPredict = model.predict(testX)
    #     # invert predictions
    #     trainPredict = scaler.inverse_transform(trainPredict)
    #     trainY = scaler.inverse_transform([trainY])
    #     testPredict = scaler.inverse_transform(testPredict)
    #     testY = scaler.inverse_transform([testY])
    #     # calculate root mean squared error
    #     trainScore = math.sqrt(metrics.mean_squared_error(trainY[0], trainPredict[:, 0]))
    #     print('Train Score: %.2f RMSE' % (trainScore))
    #     testScore = math.sqrt(metrics.mean_squared_error(testY[0], testPredict[:, 0]))
    #     print('Test Score: %.2f RMSE' % (testScore))











