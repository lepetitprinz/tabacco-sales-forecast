from Preprocessing import Preprocessing

# Importing libraries
import warnings
from typing import Dict, Callable, Any
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from math import sqrt
from sklearn.metrics import mean_squared_error

# Library of Statistical Models
# import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

# Library of Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense, LSTM

def model_stats_method(func: Callable) -> Callable[[object, list, tuple, int], Any]:
    def wrapper(obj: object, history: list, config: tuple, pred_step: int):
        return func(obj, history, config, pred_step)
    return wrapper

class ModelStats(object):
    """
    Statistical Model Class

    # Model List
    1. Univariate Model
        - AR model (Autoregressive model)
        - ARMA model (Autoregressive moving average model)
        - ARIMA model (Autoregressive integrated moving average model)
        - SES model (Simple Exponential Smoothing model
        - HWES model (Holt Winters Exponential Smooting model)

    2. Multivariate Model
        - VAR model (Vector Autoregression model)
        - VARMA model (Vector Autoregressive Moving Average model)
        - VARMAX model (Vector Autoregressive Moving Average with eXogenous regressors model)
    """
    def __init__(self):
        self.model_fit = None
        self.model_list: list = ['ar', 'arma', 'arima', 'var', 'varma', 'varmax']
        self.model: Dict[str, model_stats_method] = {'ar': self.ar_model,
                                                     'arma': self.arma_model,
                                                     'arima': self.arima_model,
                                                     'var': self.varma_model,
                                                     'varma': self.varma_model,
                                                     'varmax': self.varmax_model}

    # ------------------- #
    #  Univariate Model #
    # ------------------- #
    # Auto-regressive Model
    @model_stats_method
    def ar_model(self, history, config: tuple, pred_step=0):
        """
        :param history: time series data
        :param config:
                l: lags (int)
                t: trend ('c', 'nc')
                    c: Constant
                    nc: No Constant
                s: seasonal (bool)
                p: period (Only used if seasonal is True)
        :param pred_step: prediction steps
        :return: forecast result
        """
        t = config
        model = AR(endog=history, freq='M')
        model_fit = model.fit(trend=t)
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    # Auto regressive moving average model
    @model_stats_method
    def arma_model(self, history, config: tuple, pred_step=0):
        """
        :param history: time series data
        :param config:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                f: frequency of the time series (‘B’, ‘D’, ‘W’, ‘M’, ‘A’, ‘Q)
                t: trend ('c', 'nc')
                    c: Constant
                    nc: No constant
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = config
        # define model
        model = ARMA(endog=history, order=o, freq=f)
        # fit model
        model_fit = model.fit(trend=t, disp=0)
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    # Autoregressive integrated moving average model
    @model_stats_method
    def arima_model(self, history: list, config: tuple, pred_step=0):
        """
        :param history: time series data
        :param config:
                o: order (p, d, q)
                    p: Trend autoregression order
                    d; Trend difference order
                    q: Trend moving average order
                f: frequency of the time series (‘B’, ‘D’, ‘W’, ‘M’, ‘A’, ‘Q)
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, f, t = config
        # define model
        model = ARIMA(history, order=o, freq=f)
        # fit model
        model_fit = model.fit(trend=t, disp=0)
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat[0]

    # @model_stats_method
    # # Seasonal Autoregressive Moving Average model
    # def sarima_model(self, history:list, config: tuple, pred_step=0):
    #     """
    #     :param history: time series data
    #     :param config:
    #             o: order(p, d, q)
    #                 p: Trend autoregression order
    #                 d: Trend difference order
    #                 q: Trend moving average order
    #             t: trend ()
    #     :param pred_step: Prediction steps
    #     :return: forecast result
    #     """
    #     o, t = config
    #     model = sm.tsa.SARIMAX(history, order=o, trend=t)
    #     model_fit = model.fit()
    #     self.model_fit = model_fit
    #
    #     yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)
    #
    #     return yhat[0]

    # @model_stats_method
    # # Dynamic Factor model
    # def df_model(self, history: list, config: tuple, pred_step=0):
    #     k, o = config
    #     # define model
    #     model = sm.tsa.DynamicFactor(history, k_factors=k, factor_order=o)
    #     # fit model
    #     model_fit = model.fit()
    #     self.model_fit = model_fit
    #     # print('Coefficients: {}'.format(model_fit.params))
    #     # print(model_fit.summary())
    #     yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)
    #
    #     return yhat[0]

    # @model_stats_method
    # def holt_winters_model(self, history: list, config: tuple, pred_step=0):
    #     """
    #     :param history: time series data
    #     :param config:
    #             t: trend ('add', 'mul', 'additive', 'multiplicative')
    #                 - type of trend component
    #             d: damped_trend (bool)
    #                 - should the trend component be damped
    #             s: seasonal ('add', 'mul', 'additive', 'multiplicative', None)
    #                 - Type of seasonal component
    #             p: seasonal_periods (int)
    #                 - The number of periods in a complete seasonal cycle
    #             b: use_boxcox (True, False, ‘log’, float)
    #                 - Should the Box-Cox transform be applied to the data first?
    #             r: remove_bias (bool)
    #                 - Remove bias from forecast values and fitted values by enforcing that the average residual is equal to zero
    #     :param pred_step:
    #     :return:
    #     """
    #     t, d, s, p, b, r = config
    #     # define model
    #     model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
    #     # fit model
    #     model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)     # fit model
    #     self.model_fit = model_fit
    #     # print('Coefficients: {}'.format(model_fit.params))
    #     # print(model_fit.summary())
    #
    #     # Make multi-step forecast
    #     yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)
    #
    #     return yhat[0]

    # --------------------- #
    #  Multivariate Model #
    # --------------------- #
    @model_stats_method
    def var_model(self, history: list, pred_step=0):
        """
        :param history:
        :param config:
        :param pred_step:
        :return:
        """
        # define model
        model = VAR(history)
        # fit model
        model_fit = model.fit()
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.forecast(model_fit.y, steps=pred_step)

        return yhat

    # Vector Autoregressive Moving Average model
    def varma_model(self, history: list, config: tuple, pred_step=0):
        """
        :param history: time series data
        :param config:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, t = config
        # define model
        model = VARMAX(history,  order=o, trend=t)
        # fit model
        model_fit = model.fit(disp=False)
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat

    @model_stats_method
    # Vector Autoregressive Moving Average with eXogenous regressors model
    def varmax_model(self, history: list, data_exog: list, config: tuple, pred_step=0):
        """
        :param history: time series data
        :param data_exog: exogenous data
        :param config:
                o: order (p, q)
                    p: Trend autoregression order
                    q: Trend moving average order
                t: trend ('n', 'c', 't', 'ct')
                    n: No trend
                    c: Constant only
                    t: Time trend only
                    ct: Constant and time trend
        :param pred_step: prediction steps
        :return: forecast result
        """
        o, t = config
        # define model
        model = VARMAX(history, exog=data_exog,  order=o, trend=t)
        # fit model
        model_fit = model.fit()
        self.model_fit = model_fit
        # print('Coefficients: {}'.format(model_fit.params))
        # print(model_fit.summary())

        # Make multi-step forecast
        yhat = model_fit.predict(start=len(history), end=len(history) + pred_step)

        return yhat

    def walk_forward_validation(self, model: str, data, n_test, config, pred_step):
        """
        :param model: Statistical model
                'ar': Autoregressive model
                'arma': Autoregressive Moving Average model
                'arima': Autoregressive Integrated Moving Average model
                'varmax': Vector Autoregressive Moving Average with eXogenous regressors model
                'hw': Holt Winters model
        :param data:
        :param n_test: number of test data
        :param config: configuration
        :param pred_step: prediction steps
        :return:
        """
        predictions = list()

        # split dataset
        train, test = self.train_test_split(data=data, n_test=n_test)
        history = [x for x in train]  # seed history with training dataset

        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.model[model](history=history,
                                     config=config,
                                     pred_step=n_test-i)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])

        # estimate prediction error
        error = self.calc_sqrt_mse(test, predictions)
        return error

    def score_model(self, model: str, data, n_test, config, pred_step: int):
        result = None

        # convert config to a key
        key = str(config)
        result = self.walk_forward_validation(model=model, data=data, n_test=n_test,
                                              config=config, pred_step=pred_step)
        return (model, key, result)

    def grid_search(self, model: str, data, n_test: int, pred_step: int, cfg_list: list):
        scores = [self.score_model(model=model, data=data, n_test=n_test,
                                   config=config, pred_step=pred_step) for config in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[2])

        return scores

    def train_test_split(self, data, n_test):
        return data[:-n_test], data[-n_test:]

    def calc_sqrt_mse(self, actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

class ModelDeepLearning(object):
    """
    LSTM
        - Data format: (samples, timesteps, features)
            - samples:
            - timesteps: used as input
            - features: data features used in training

    """
    # Deep Learning Hyperparameters
    LSTM_SIMPLE_UNIT = 32
    EPOCHS = 10
    BATCH_SIZE = 32

    def __init__(self):
        self.preprocessing = Preprocessing()
        self.history = None

    def lstml_vanilla(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                      units: int, timesteps: int, pred_steps=1):

        x_train, y_train = self.preprocessing.split_sequence_univ(df=train, feature='Close',
                                                                  timesteps=timesteps, pred_steps=pred_steps)
        x_val, y_val = self.preprocessing.split_sequence_univ(df=val, feature='Close',
                                                              timesteps=timesteps, pred_steps=pred_steps)
        x_test, y_test = self.preprocessing.split_sequence_univ(df=test, feature='Close',
                                                                timesteps=timesteps, pred_steps=pred_steps)
        # Reshape
        n_features = 1
        x_train = self.lstm_data_reshape(data=x_train, n_feature=n_features)
        x_val = self.lstm_data_reshape(data=x_val, n_feature=n_features)
        x_test = self.lstm_data_reshape(data=x_test, n_feature=n_features)

        # Build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(timesteps, n_features)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mae')

        history = model.fit(x_train, y_train,
                            epochs=self.__class__.EPOCHS,
                            batch_size=self.__class__.BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            shuffle=False)
        self.history = history

        predictions = model.predict(x_test, verbose=0)
        rmse = self.calc_sqrt_mse(actual=y_test, predicted=predictions)
        print('Test RMSE: %.3f' % rmse)

        return predictions, rmse

    def lstml_stacked(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                      units: int, timesteps: int, pred_steps=1):

        x_train, y_train = self.preprocessing.split_sequence_univ(df=train, feature='Close',
                                                                  timesteps=timesteps, pred_steps=pred_steps)
        x_val, y_val = self.preprocessing.split_sequence_univ(df=val, feature='Close',
                                                              timesteps=timesteps, pred_steps=pred_steps)
        x_test, y_test = self.preprocessing.split_sequence_univ(df=test, feature='Close',
                                                                timesteps=timesteps, pred_steps=pred_steps)
        # reshape data
        n_features = 1
        x_train = self.lstm_data_reshape(data=x_train, n_feature=n_features)
        x_val = self.lstm_data_reshape(data=x_val, n_feature=n_features)
        x_test = self.lstm_data_reshape(data=x_test, n_feature=n_features)

        # build model
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', return_sequences=True, input_shape=(timesteps, n_features)))
        model.add(LSTM(units=units, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='mae')

        # fit model
        history = model.fit(x_train, y_train,
                            epochs=self.__class__.EPOCHS,
                            batch_size=self.__class__.BATCH_SIZE,
                            validation_data=(x_val, y_val),
                            shuffle=False)
        self.history = history

        predictions = model.predict(x_test, verbose=0)
        rmse = self.calc_sqrt_mse(actual=y_test, predicted=predictions)
        print('Test RMSE: %.3f' % rmse)

        return predictions, rmse

    def lstm_data_reshape(self, data: np.array, n_feature: int):
        return data.reshape((data.shape[0], data.shape[1], n_feature))

    def calc_sqrt_mse(self, actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))
