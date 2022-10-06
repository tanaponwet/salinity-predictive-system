# -*- coding: utf-8 -*-
"""
@author: tanap

"""
#%% import library(s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import mean_squared_error

plt.style.use('dark_background')

#%% read csv file
df = pd.read_csv('data/BTC-USD.csv')

#%%
df_original = df.filter(['Date','Open'])
print(df_original.dtypes)

#%%
df_original['Date'] = pd.to_datetime(df_original['Date'])
print(df_original.dtypes)

#%%
df_original.set_index('Date', inplace=True)

#%%
df_original = df_original.asfreq(freq='D', method='ffill')

#%% visualize Open Price History
plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.plot(df_original)
plt.show()

#%%
from statsmodels.tsa.stattools import adfuller
adf, pvalue, usedlag_, nobs_, critical_values_, icbest_ = adfuller(df_original)
print("pvalue = ", pvalue, " if above 0.05, data is not stationary")

#%%
from statsmodels.tsa.seasonal import seasonal_decompose 
decomposed = seasonal_decompose(df_original['Open'],  
                            model ='additive')

#%%
trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

#%%
plt.figure(figsize=(16,8))
plt.subplot(411)
plt.plot(df_original, label='Original', color='yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label='Seasonal', color='yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residual', color='yellow')
plt.legend(loc='upper left')
plt.show()

#%%
# =============================================================================
# Split data to Train, Test
# =============================================================================

#%%
train_set_len = math.ceil(len(df_original) * 0.80)

#%%
train_set = list(df_original[0:train_set_len]['Open'])
test_set = list(df_original[train_set_len:]['Open'])

#%%
plt.figure(figsize=(16,8))
plt.grid(True)
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.plot(df_original[0:train_set_len]['Open'], label='Train set')
plt.plot(df_original[train_set_len:]['Open'], label='Test set')
plt.legend()
plt.show()

#%%
# =============================================================================
# ARIMA Model
# =============================================================================

#%% import library(s)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

#%%
predictions = []
n_test = len(test_set)

#%%
for i in range(n_test):
    model = ARIMA(train_set, order=(4,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = list(output)[0]
    predictions.append(yhat)
    actual_test_value = test_set[i]
    train_set.append(actual_test_value)
    # print(output)
    # print(yhat)
    # break

#%%
print(model_fit.summary())

#%%
plt.figure(figsize=(16,8))
plt.grid(True)

date_range = df_original[train_set_len:].index

plt.plot(date_range, predictions, color='blue', marker = 'o', linestyle = 'dashed', label='Predited Open')
plt.plot(date_range, test_set, color='red', marker = 'o', linestyle = 'dashed', label='Actual Open')
plt.title('Predited Open vs. Actual Open')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#%%
plt.figure(figsize=(16,8))
plt.grid(True)

date_range = df_original[train_set_len:].index

plt.plot(date_range, predictions, label='Predited Open')
plt.plot(date_range, test_set, label='Actual Open')
plt.title('Predited Open vs. Actual Open')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

#%% get the root mean squared error (RSME)
rmse = mean_squared_error(test_set, predictions, squared=False)
print(rmse)

#%%
# =============================================================================
# Auto ARIMA (not used)
# =============================================================================

#%% import library
from pmdarima.arima import auto_arima

#%%
arima_model = auto_arima(df_original[0:train_set_len]['Open'],
                          max_p = 5, max_d = 5, max_q = 5,
                          seasonal = False, 
                          trace = True, 
                          error_action ='warn',   
                          suppress_warnings = False,  
                          stepwise = True, n_fits=50)           

#%%
print(arima_model.summary())

