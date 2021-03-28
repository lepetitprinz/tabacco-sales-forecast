import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from Preprocessing import Preprocessing
from TimeSeriesAnalysis import TimeSeriesAnalysis
from Model_BAK import ModelStats, ModelDeepLearning
from Forecast import Forecast
# from Simulation import Simulation

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 1. Preprocessing
# -*-*-*-*-*-*-*-*-*-*-*-*- #

# 1.1 Load raw data
preprocessing = Preprocessing()

model_2_nm = 'model_2_dfs.csv'

# Load dataset
df = preprocessing.load_data(file_path=os.path.join('..', 'input', model_2_nm))

# Convert type: string to datetime
df = preprocessing.conv_to_datetime(df=df, feature='yyyymm')

# Convert date column to index
df = preprocessing.date_to_idx(df=df, feature='yyyymm')   # 'Date' column: date data

# View statistics of dataset
preprocessing.view_stats(df=df)

# Check missing values
preprocessing.check_na(df=df)

# Fill missing values
df = preprocessing.fill_na(df=df)

# Conver to float scale
df = df * 1.0

# Visualization by Date Scale
# .veiw_plot_as_freq(df=df, feature='TOT', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=df, feature='TOT', freq='Q', figsize=(8, 4), folder_nm='model_2')    # Quarter
preprocessing.veiw_plot_as_freq(df=df, feature='TOT', freq='M', figsize=(8, 4), folder_nm='model_2')    # Month

# preprocessing.veiw_plot_as_freq(df=df, feature='GR', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=df, feature='GR', freq='Q', figsize=(8, 4), folder_nm='model_2')    # Quarter
preprocessing.veiw_plot_as_freq(df=df, feature='GR', freq='M', figsize=(8, 4), folder_nm='model_2')    # Month

# preprocessing.veiw_plot_as_freq(df=df, feature='NGP', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=df, feature='NGP', freq='Q', figsize=(8, 4), folder_nm='model_2')    # Quarter
preprocessing.veiw_plot_as_freq(df=df, feature='NGP', freq='M', figsize=(8, 4), folder_nm='model_2')    # Month

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 2. Time Series Analysis
# -*-*-*-*-*-*-*-*-*-*-*-*- #

tsa = TimeSeriesAnalysis()


tsa.view_corr_plot(df=df, folder_nm='model_2')

# Plot Rate Change
tsa.view_rate_change_plot(df=df, feature='TOT', folder_nm='model_2')
tsa.view_rate_change_plot(df=df, feature='GR', folder_nm='model_2')
tsa.view_rate_change_plot(df=df, feature='NGP', folder_nm='model_2')

# Plot trend, seasonality, residual
tsa.view_seasonal_decompose_plot(df=df, feature='TOT', folder_nm='model_2')
tsa.view_seasonal_decompose_plot(df=df, feature='GR', folder_nm='model_2')
tsa.view_seasonal_decompose_plot(df=df, feature='NGP', folder_nm='model_2')

# Check auto-correlation
auto_corr_tot = tsa.check_auto_correlation(df=df, feature='TOT', lag=1)
auto_corr_gr = tsa.check_auto_correlation(df=df, feature='GR', lag=1)
auto_corr_ngp = tsa.check_auto_correlation(df=df, feature='NGP', lag=1)

# Check stationarity
stationarity_tot = tsa.check_stationarity(df=df, feature='TOT', conf_lvl=0.05)
stationarity_gr = tsa.check_stationarity(df=df, feature='GR', conf_lvl=0.05)
stationarity_ngp = tsa.check_stationarity(df=df, feature='NGP', conf_lvl=0.05)

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 3. Model
# -*-*-*-*-*-*-*-*-*-*-*-*- #

# ----------------------- #
# 3.1 Statistical Model
# ----------------------- #
model_stats = ModelStats()

# # Configurations of each model
lag = 4                 # AR model
seasonal = True         # AR model
trend = 'ct'            # AR / VARMAX / HW model
trend_arma = 'c'        # ARMA model
period = 12             # AR / HW model
frequency = None        # ARMA / ARIMA model
two_lvl_order = (1, 0)  # ARMA / VARMAX model
three_lvl_order = (1, 0, 1)    # ARIMA model

trend_hw = 'add'
damped_trend = True     # Holt-Winters model
seasonal_mtd = 'add'    # Holt-Winters model
use_boxcox = False      # Holt-Winters model
remove_bias = True      # Holt-Winters model

config_ar = (lag, trend, seasonal, period)
config_arma = (two_lvl_order, frequency, trend_arma)
config_arima = (three_lvl_order, frequency, trend_arma)
config_varmax = (two_lvl_order, trend)
config_hw = (trend_hw, damped_trend, seasonal_mtd, period, use_boxcox, remove_bias)


error_ar = model_stats.walk_forward_validation(model='ar', data=data, n_test=n_test,
                                               config=config_ar, pred_step=pred_step)
error_arma = model_stats.walk_forward_validation(model='arma', data=data, n_test=n_test,
                                                 config=config_arma, pred_step=pred_step)
# error_arima = model_stats.walk_forward_validation(model='arima', data=data, n_test=n_test,
#                                                   config=config_arima, pred_step=pred_step)
# error_varmax = model_stats.walk_forward_validation(model='varmax', data=data, n_test=n_test,
#                                                    config=config_varmax, pred_step=pred_step)
error_hw = model_stats.walk_forward_validation(model='hw', data=data, n_test=n_test,
                                               config=config_hw, pred_step=pred_step)

error = pd.DataFrame({'model': ['AR', 'ARMA', 'HW'],
                      'error': [error_ar, error_arma, error_hw]})
error = error.sort_values(by='error', axis=0, inplace=False)


