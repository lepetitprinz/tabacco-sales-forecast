import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')    # Time series style

from sklearn.model_selection import TimeSeriesSplit

class Preprocessing(object):
    ROOT_DIRECTORY: str = os.path.join('..', 'input')

    # Split rate
    VAL_SPLIT_RATE = 0.2
    TEST_SPLIT_RATE = 0.1
    NESTED_CV_SPLIT = 5  # Split Time Series
    # IMG_SAVE_PATH = os.path.join('http://ba.bimatrix.co.kr', 'matrix', 'ktng', 'model')
    IMG_SAVE_PATH = os.path.join('home', 'matrix6', 'apps', 'tomcat-8.5.39', 'webapps', 'ktng')

    def __init__(self):
        self.check_dir()

    def check_dir(self):
        if not os.path.exists(self.ROOT_DIRECTORY):
            os.mkdir(self.ROOT_DIRECTORY)

    def load_data(self, file_path: str):
        """
        :return: time series data
        """
        df = pd.read_csv(os.path.join(self.ROOT_DIRECTORY, file_path), sep='\t')
        return df

    def conv_to_datetime(self, df: pd.DataFrame, feature: str, format='%Y%m'):
        df[feature] = pd.to_datetime(df[feature], format=format)
        return df

    def date_to_idx(self, df: pd.DataFrame, feature: str):
        df = df.set_index(feature)
        return df

    def view_stats(self, df: pd.DataFrame):
        print(df.describe().transpose())

    # Count missing values of the each feature
    def check_na(self, df: pd.DataFrame):
        print('Missing values counts of the each feature')
        print(df.isna().sum())

    # Fill missing values(forward/backward filling)
    def fill_na(self, df: pd.DataFrame, method='ffill'):
        """
        :param df: time series data
        :param method: ffill / bfill
        """
        df = df.fillna(method=method)
        print('Filling missing values is finished')
        return df

    # Remove missing values
    def filtter_na(self, df: pd.DataFrame, drop_col: list):
        """
        :param df: time series data
        :param drop_col: drop columns
        """
        if len(drop_col) != 0:
            df = df.dropna(how='any', subset=drop_col)
        else:
            df = df.dropna(how='any')
        return df

    def def_date_range(self, start: str, end: str, freq='D'):
        """~
        :param start: start date    ex) '1/1/20'
        :param end: end date        ex) '1/1/21'
        :param freq:  'H', 'D' , 'M' , 'Q' , 'Y'
        :return:
        """
        date_range = pd.date_range(start=start, end=end, freq=freq)
        return date_range

    # Visualize the plot on frequency
    def veiw_plot_as_freq(self, df: pd.DataFrame, feature: str, folder_nm: str, freq='M', figsize=(6, 4)):
        """
        :param df: time series data
        :param feature: visualizing column
        :param freq: 'H', 'D' , 'M' , 'Q' , 'Y'
        :param figsize: figure size
        """
        plt.clf()
        df[feature].asfreq(freq, method='ffill').plot(figsize=figsize, linewidth=0.7)
        plt.title(f"Frequency plot - freq: {freq} / feature: {feature}", fontsize=12)
        plt.savefig(os.path.join('img', folder_nm, 'plot_as_freq_' + feature + '_' + freq + '.png'))

    # Visualize date range plot
    def veiw_date_rng_plot(self, df: pd.DataFrame, from_date, to_date, folder_nm: str, figsize=(8, 6)):
        """
        :param df: time series data
        :param from_date: start date
        :param to_date: end date
        :param figsize: figure size
        """
        plt.clf()
        df[from_date: to_date].plot(subplots=True, figsize=figsize,  linewidth=0.7)
        plt.savefig(os.path.join('img', folder_nm, 'plot_' + from_date + '_' + to_date + '.png'))

    # Shift date on dafaframe
    def date_shift_of_df(self, df: pd.DataFrame, shift_period: int, freq='D'):
        """
        :param df: time series data
        :param shift_period: shift period
        :param freq: 'H', 'D' , 'M' , 'Q' , 'Y'
        :return:
        """
        shifted = df.asfreq(freq=freq).shift(periods=shift_period)
        return shifted

    # Shift date on dafaframe
    def date_shift_of_feat(self, df: pd.DataFrame, feature: str, shift_period: int, freq='D'):
        shifted = df[feature].asfreq(freq=freq).shift(periods=shift_period)
        return shifted

    def resampling(self, df: pd.DataFrame, method: str, rule: str, agg_method='mean'):
        """
        :param df: time series data
        :param method: 'up' / 'down'
        :param rule: 'H', 'D' , 'M' , 'Q' , 'Y'
        :param agg_method: 'mean', 'median'
        """
        if method == 'up':
            if agg_method == 'mean':
                df = df.resample(rule=rule).mean()
            elif agg_method == 'median':
                df = df.resample(rule=rule).median()

        elif method == 'down':
            df = df.resample(rule=rule).pad()

        return df

    def split_stat_data(self, df: pd.DataFrame):
        n = len(df)
        train_df = df[0: int(n*(1-self.TEST_SPLIT_RATE))]
        test_df = df[int(n*self.TEST_SPLIT_RATE):]

        return train_df, test_df

    def split_data_walk_forward(self, df: pd.DataFrame):
        n = len(df.values)
        train_split_rate = 1 - self.__class__.VAL_SPLIT_RATE - self.__class__.TEST_SPLIT_RATE
        train_df = df.iloc[:int(n*train_split_rate)]
        val_df = df.iloc[int(n*train_split_rate): int(n*(1-self.__class__.TEST_SPLIT_RATE))]
        test_df = df.iloc[int(n*(1-self.__class__.TEST_SPLIT_RATE)):]

        return train_df, val_df, test_df

    def split_data_nested_cross_validation(self, df: pd.DataFrame):
        splits = TimeSeriesSplit(n_splits=self.NESTED_CV_SPLIT)
        data = df.values
        idx = df.index

        train_idx_set = []
        train_cv_set = []
        test_idx_set = []
        test_cv_set = []
        for train_idx, test_idx in splits.split(data):
            train_idx_set.append(idx[train_idx])
            train_cv_set.append(data[train_idx])
            test_idx_set.append(idx[test_idx])
            test_cv_set.append(data[test_idx])

        return train_idx_set, train_cv_set, test_idx_set, test_cv_set

    def normalize_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df_scaled = (train_df - train_mean) / train_std
        val_df_scaled = (val_df - train_mean) / train_std
        test_df_scaled = (test_df - train_mean) / train_std

        return train_df_scaled, val_df_scaled, test_df_scaled

    def view_norm_dist(self, df: pd.DataFrame, train_df: pd.DataFrame):
        df_std = (df - train_df.mean()) / train_df.std()
        labels = list(df_std.columns)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        ax.violinplot(dataset=df_std.values)
        ax.set_xticks(np.arange(1, len(labels)+1))
        ax.set_xticklabels(labels)
        plt.savefig(os.path.join('img', 'plot_normalized_distribution.png'))

    def split_sequence_univ(self, df: pd.DataFrame, feature: str, timesteps: int, pred_steps=1):
        """
        Split univariate sequence data
        :param df: Time series data
        :param timesteps:
        :return:
        """
        data = df[feature].values
        n = len(data)

        X, y = list(), list()
        for i in range(n):
            # find the end of this pattern
            end_ix = i + timesteps
            pred_ix = end_ix + pred_steps
            # check if we are beyond the sequence
            if end_ix > n - 1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix], data[end_ix:pred_ix]
            X.append(seq_x)
            y.append(seq_y)

        return np.array(X), np.array(y)