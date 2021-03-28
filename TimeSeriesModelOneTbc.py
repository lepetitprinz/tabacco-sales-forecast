import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from Preprocessing import Preprocessing
from TimeSeriesAnalysis import TimeSeriesAnalysis
from Model import ModelStats, ModelDeepLearning
from Forecast import Forecast
# from Simulation import Simulation

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 1. Preprocessing
# -*-*-*-*-*-*-*-*-*-*-*-*- #

# 1.1 Load raw data
preprocessing = Preprocessing()

tbc = 'TBC.csv'

# Load dataset
tbc = preprocessing.load_data(file_path=os.path.join('..', 'input', tbc))

# Convert type: string to datetime
tbc = preprocessing.conv_to_datetime(df=tbc, feature='YYYYMM')

# Convert date column to index
tbc = preprocessing.date_to_idx(df=tbc, feature='YYYYMM')   # 'Date' column: date data

# View statistics of dataset
preprocessing.view_stats(df=tbc)

# Check missing values
preprocessing.check_na(df=tbc)

# Fill missing values
tbc = preprocessing.fill_na(df=tbc)

# Conver to float scale
tbc = tbc * 1.0

# Visualization by Date Scale
preprocessing.veiw_plot_as_freq(df=tbc, feature='TOT', freq='Y', figsize=(8, 4), folder_nm='model_1')    # Year
preprocessing.veiw_plot_as_freq(df=tbc, feature='TOT', freq='Q', figsize=(8, 4), folder_nm='model_1')    # Quarter
preprocessing.veiw_plot_as_freq(df=tbc, feature='TOT', freq='M', figsize=(8, 4), folder_nm='model_1')    # Month

# preprocessing.veiw_plot_as_freq(df=df, feature='GR', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=tbc, feature='GR', freq='Q', figsize=(8, 4), folder_nm='model_1')    # Quarter
preprocessing.veiw_plot_as_freq(df=tbc, feature='GR', freq='M', figsize=(8, 4), folder_nm='model_1')    # Month

# preprocessing.veiw_plot_as_freq(df=df, feature='NGP', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=tbc, feature='NGP', freq='Q', figsize=(8, 4), folder_nm='model_1')    # Quarter
preprocessing.veiw_plot_as_freq(df=tbc, feature='NGP', freq='M', figsize=(8, 4), folder_nm='model_1')    # Month

# # -*-*-*-*-*-*-*-*-*-*-*-*- #
# # 2. Time Series Analysis
# # -*-*-*-*-*-*-*-*-*-*-*-*- #

tsa = TimeSeriesAnalysis()

# Plot correlation of features
tsa.view_corr_plot(df=tbc, folder_nm='model_1')

# Plot Rate Change
tsa.view_rate_change_plot(df=tbc, feature='TOT', folder_nm='model_1')
tsa.view_rate_change_plot(df=tbc, feature='GR', folder_nm='model_1')
tsa.view_rate_change_plot(df=tbc, feature='NGP', folder_nm='model_1')

# Plot trend, seasonality, residual
tsa.view_seasonal_decompose_plot(df=tbc, feature='TOT', folder_nm='model_1')
tsa.view_seasonal_decompose_plot(df=tbc, feature='GR', folder_nm='model_1')
tsa.view_seasonal_decompose_plot(df=tbc, feature='NGP', folder_nm='model_1')

# Check auto-correlation
auto_corr_tot = tsa.check_auto_correlation(df=tbc, feature='TOT', lag=1)
auto_corr_gr = tsa.check_auto_correlation(df=tbc, feature='GR', lag=1)
auto_corr_ngp = tsa.check_auto_correlation(df=tbc, feature='NGP', lag=1)

# Check stationarity
stationarity_tot = tsa.check_stationarity(df=tbc, feature='TOT', conf_lvl=0.05)
stationarity_gr = tsa.check_stationarity(df=tbc, feature='GR', conf_lvl=0.05)
stationarity_ngp = tsa.check_stationarity(df=tbc, feature='NGP', conf_lvl=0.05)

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 3. Model
# -*-*-*-*-*-*-*-*-*-*-*-*- #

# Data (Univariate Dataset)
tbc_gr = tbc['GR']
tbc_npg = tbc['NGP']

# ----------------------- #
# 3.1 Statistical Model
# ----------------------- #
model_stats = ModelStats()

# Set configuration of statistical model
n_test = 3      # Test period
pred_step = 5   # Forecast period

# # Configurations of each model
trend = 'c'            # AR
trend_arma = 'c'        # ARMA model
frequency = None        # ARMA / ARIMA model
two_lvl_order = (1, 3)  # ARMA
three_lvl_order = (1, 0, 3)    # ARIMA model

# Holt-Winters Configuration
# trend_hw = 'add'        # Holt-Winters model
# damped_trend = True     # Holt-Winters model
# seasonal_mtd = 'add'    # Holt-Winters model
# use_boxcox = False      # Holt-Winters model
# remove_bias = True      # Holt-Winters model

config_ar = (trend)
config_arma = (two_lvl_order, frequency, trend_arma)
config_arima = (three_lvl_order, frequency, trend_arma)
config_varmax = (two_lvl_order, trend)
# config_hw = (trend_hw, damped_trend, seasonal_mtd, period, use_boxcox, remove_bias)

erro_gr_ar = model_stats.walk_forward_validation(model='ar', data=tbc_gr, n_test=n_test,
                                               config=config_ar, pred_step=pred_step)

error_gr_arma = model_stats.walk_forward_validation(model='arma', data=tbc_gr, n_test=n_test,
                                                 config=config_arma, pred_step=pred_step)
error_arima = model_stats.walk_forward_validation(model='arima', data=tbc_gr, n_test=n_test,
                                                  config=config_arima, pred_step=pred_step)

error_gr = pd.DataFrame({'model': ['AR', 'ARMA', 'ARIMA'],
                      'error': [erro_gr_ar, error_gr_arma, error_arima]})
error = error_gr.sort_values(by='error', axis=0, inplace=False)


print(error)

