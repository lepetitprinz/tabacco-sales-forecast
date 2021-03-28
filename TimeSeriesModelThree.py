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

df = 'model_3.csv'

# Load dataset
df = preprocessing.load_data(file_path=os.path.join('..', 'input', df))

df = preprocessing.conv_to_datetime(df=df, feature='yyyymm')

# Convert date column to index
df = preprocessing.date_to_idx(df=df, feature='yyyymm')   # 'Date' column: date data

df = df[['GR_DOM', 'NGP_DOM', '_0', '_1','_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10']]

tsa = TimeSeriesAnalysis()
tsa.view_corr_plot(df=df, folder_nm='model_3')
print('')
