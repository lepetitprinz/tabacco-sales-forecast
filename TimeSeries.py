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

test = 'GOOGL_2006-01-01_to_2018-01-01.csv'

# Load dataset
# df = preprocessing.load_data(file_path=os.path.join('..', 'input', test))
df = pd.read_csv(os.path.join('..', 'input', test))

# Drop unnecessary columns
# df = df.drop(columns=['Name'], axis=1)

# Convert type: string to datetime
df = preprocessing.conv_to_datetime(df=df, feature='Date', format='%Y-%m-%d')

# Convert date column to index
df = preprocessing.date_to_idx(df=df, feature='Date')   # 'Date' column: date data

# View statistics of dataset
preprocessing.view_stats(df=df)

# Check missing values
preprocessing.check_na(df=df)

# Fill missing values
df = preprocessing.fill_na(df=df)

# Define the time range
date_range = preprocessing.def_date_range(start='7/1/2017', end='7/31/2020', freq='M')

# Visualization by Date Scale
preprocessing.veiw_plot_as_freq(df=df, feature='Close', freq='Y', figsize=(8, 4))    # Year
preprocessing.veiw_plot_as_freq(df=df, feature='Close', freq='Q', figsize=(8, 4))    # Quarter
preprocessing.veiw_plot_as_freq(df=df, feature='Close', freq='M', figsize=(8, 4))    # Month
preprocessing.veiw_plot_as_freq(df=df, feature='Close', freq='D', figsize=(8, 4))    # Day

# Visualization on specific data range
preprocessing.veiw_date_rng_plot(df=df, from_date='2017-07-01', to_date='2017-12-31')  # 2017 year
# preprocessing.veiw_date_rng_plot(df=df, from_date='2018-01-01', to_date='2018-12-31')  # 2018 year
# preprocessing.veiw_date_rng_plot(df=df, from_date='2019-01-01', to_date='2019-12-31')  # 2018 year
# preprocessing.veiw_date_rng_plot(df=df, from_date='2020-01-01', to_date='2020-07-31')   # 2019 year

# shifted_month_df = preprocessing.date_shift_of_df(df=df, shift_period=6, freq='M')
# shifted_month_sales = preprocessing.date_shift_of_feat(df=df, feature='sales', shift_period=6, freq='M')

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 2. Time Series Analysis
# -*-*-*-*-*-*-*-*-*-*-*-*- #

tsa = TimeSeriesAnalysis()

# Plot Rate Change
tsa.view_rate_change_plot(df=df, feature='Close')

# Plot trend, seasonality, residual
tsa.view_seasonal_decompose_plot(df=df, feature='Close')

# Check auto-correlation
auto_corr = tsa.check_auto_correlation(df=df, feature='Close', lag=1)

# Check stationarity
stationarity = tsa.check_stationarity(df=df, feature='Close', conf_lvl=0.05)

# -*-*-*-*-*-*-*-*-*-*-*-*- #
# 3. Model
# -*-*-*-*-*-*-*-*-*-*-*-*- #

# ----------------------- #
# 3.1 Statistical Model
# ----------------------- #
model_stats = ModelStats()

# Choose univariate dataset
data = np.array(df['Close'])

# Set configuration of statistical model
n_test = 1
pred_step = 0

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

# Statistical Learning using walk forward validation
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

# ----------------------- #
# 3.2 Deep Learning Model
# ----------------------- #

# Split data used in deep learning
train_df, val_df, test_df = preprocessing.split_data_walk_forward(df=df)
# Normalize data
train_df_scaled, val_df_scaled, test_df_scaled = preprocessing.normalize_data(train_df=train_df,
                                                                              val_df=val_df,
                                                                              test_df=test_df)
# Make model
model_dl = ModelDeepLearning()
predictions_vnl, rmse_vnl = model_dl.lstml_vanilla(train=train_df, val=val_df, test=test_df,
                       units=16, timesteps=5, pred_steps=1)

predictions_stk, rmse_stk = model_dl.lstml_stacked(train=train_df, val=val_df, test=test_df,
                       units=16, timesteps=5, pred_steps=1)