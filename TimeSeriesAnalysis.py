import os
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')    # Time series style
# plt.style.use('fivethirtyeight')    # Time series style
from pylab import rcParams
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

class TimeSeriesAnalysis(object):

    def __init__(self):
        pass

    def view_corr_plot(self, df: pd.DataFrame, folder_nm: str):
        corr = np.array(df.corr().values)
        corr = np.round(corr, 2)
        fig, ax = plt.subplots()
        ax.grid(False)
        im = ax.imshow(corr, cmap='RdBu')

        ax.set_xticks(np.arange(len(corr)))
        ax.set_yticks(np.arange(len(corr)))

        ax.set_xticklabels(list(df.columns))
        ax.set_yticklabels(list(df.columns))

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(corr)):
            for j in range(len(corr)):
                text = ax.text(j, i, corr[i, j], ha="center", va="center", color="w")
        fig.tight_layout()
        plt.show()
        plt.savefig(os.path.join('img', folder_nm, 'corr_heatmap.png'))

    # Visualize rate changing graph
    def view_rate_change_plot(self, df: pd.DataFrame, feature: list, folder_nm: str,):
        rate_change = df[feature].div(df[feature].shift(1))
        plt.clf()
        rate_change.plot(linewidth=0.7)
        plt.title(f"Rate Change Plot: '{feature}' feature", fontsize=12)
        plt.savefig(os.path.join('img', folder_nm, 'plot_rate_change' + '_' + feature + '.png'))

    # Visualize trend, seasonality, residual graph
    def view_seasonal_decompose_plot(self, df: pd.DataFrame, feature: str, folder_nm: str):
        rcParams['figure.figsize'] = 11, 9
        rcParams['lines.linewidth'] = 0.7
        rcParams['lines.markersize'] = 0.7
        # decomposed_feat = seasonal_decompose(np.array(df[feature]), model='additive', freq='M')
        decomposed_feat = seasonal_decompose(df[feature], model='additive')
        plt.clf()
        figure = decomposed_feat.plot()
        plt.savefig(os.path.join('img', folder_nm, 'seasonal_decompose_plot' + '_' + feature + '.png'))

    # Check the auto-correlation
    def check_auto_correlation(self, df: pd.DataFrame, feature: str, lag=1):
        auto_corr = df[feature].autocorr(lag=lag)
        print(f'Auto correlation of column {feature} : {auto_corr:4.4f}')
        return auto_corr

    # Check the stationarity
    def check_stationarity(self, df: pd.DataFrame, feature: str, conf_lvl=0.05):
        test = adfuller(df[feature])    # Augmented Dickey-Fuller test
        result = test[0] < conf_lvl
        if result:
            print(f"Stationarity of column '{feature}' exist")
        else:
            print(f"Stationarity of column '{feature}' dose not exist")
        return result

    # def check_random_walk(self, df: pd.DataFrame, feature: str):
    #     plt.plot(df[feature])