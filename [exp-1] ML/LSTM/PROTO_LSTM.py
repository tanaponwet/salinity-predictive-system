# -*- coding: utf-8 -*-
"""
@author: tanap

dataset: https://finance.yahoo.com/quote/GE/history/
"""
#%% import library(s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, LSTM
# from keras.layers import CuDNNLSTM
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner
from keras_tuner.tuners import BayesianOptimization, Hyperband

#%% read csv file
df = pd.read_csv('data/GE.csv')

#%% clone df
df_original = df

#%% convert 'Date' to datetime
df_original['Date'] = pd.to_datetime(df_original['Date'])

#%% visualize Open Price History
plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.plot(df_original['Date'], df_original['Open'])
plt.show()

#%%
# =============================================================================
# Split data to Train, Validation, Test
# =============================================================================

#%% select argument(s)
data = df.filter(['Open','High','Low','Close','Adj Close'])
# data = df.filter(['Open'])

#%% convert pandas dataframe to numpy array
dataset = data.values

#%%
train_set_len = math.ceil(len(dataset) * 0.80)
validation_set_len = math.ceil(len(dataset) * 0.10)

#%% normalize dataset
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(dataset)

#%%
n_future = 1
n_past = 14

#%%
# =============================================================================
# Train set
# =============================================================================

#%% create Train set
train_set = scaled_dataset[0:train_set_len, :]

x_train = []
y_train = []

for i in range(n_past, len(train_set) - n_future +1):
    x_train.append(train_set[i - n_past:i, 0:dataset.shape[1]])
    y_train.append(train_set[i + n_future - 1:i + n_future, 0])
    
#%% convert Train set to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#%% reshape x_train if dim equal 2
if x_train.ndim == 2:
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
    
#%%
# =============================================================================
# Validation set
# =============================================================================

#%% create Validation set
validation_set = scaled_dataset[train_set_len - n_past:train_set_len + validation_set_len, :]

x_valid = []
y_valid = []

for i in range(n_past, len(validation_set) - n_future +1):
    x_valid.append(validation_set[i - n_past:i, 0:dataset.shape[1]])
    y_valid.append(validation_set[i + n_future - 1:i + n_future, 0])
    
#%% convert Validation set to numpy array
x_valid, y_valid = np.array(x_valid), np.array(y_valid)

#%% reshape x_valid if dim equal 2
if x_valid.ndim == 2:
    x_valid = np.reshape(x_valid, (x_valid.shape[0],x_valid.shape[1], 1))
    
#%%
# =============================================================================
# Test set
# =============================================================================    

#%% create Test set
test_set = scaled_dataset[(train_set_len + validation_set_len) - n_past:, :]

x_test = []
y_test = dataset[train_set_len + validation_set_len:, 0]

for i in range(n_past, len(test_set)):
    x_test.append(test_set[i - n_past:i, 0:dataset.shape[1]])
    
#%% convert data to a numpy array
x_test = np.array(x_test)

#%% reshape x_test if dim equal 2
if x_test.ndim == 2:
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

#%%
plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.plot(df_original[0:train_set_len]['Date'],df_original[0:train_set_len]['Open'])
plt.plot(df_original[train_set_len - n_past:train_set_len + validation_set_len]['Date'],df_original[train_set_len - n_past:train_set_len + validation_set_len]['Open'])
plt.plot(df_original[(train_set_len + validation_set_len) - n_past:]['Date'],df_original[(train_set_len + validation_set_len) - n_past:]['Open'])
plt.legend(['Train set','Validation set','Test set'], loc='upper right')
plt.show()

#%%
# =============================================================================
# Build Model
# =============================================================================

#%% build model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(16))
model.add(Dense(1))

#%% complie model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

#%%
early_stopped = EarlyStopping(monitor='val_loss', patience=10)

#%%
checkpoint_filepath = 'checkpoint/checkpoint.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch',
    verbose=1
    )

#%% train model
model_fit = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), callbacks=[model_checkpoint_callback, early_stopped])

#%% visualize evaluation
plt.figure(figsize=(16,8))
plt.title('Loss rate')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.plot(model_fit.history['loss'], label='Training loss')
plt.plot(model_fit.history['val_loss'], label='Validation loss')
plt.legend(['Training loss','Validation loss'], loc='upper right')
plt.show()

#%% predict open price
pred = model.predict(x_test)

#%%
if dataset.shape[1] > 1:
    pred = np.repeat(pred, dataset.shape[1], axis=-1)

#%% denormalize predicted values
predictions = scaler.inverse_transform(pred)[:,0]

#%% get the root mean squared error (RSME)
rmse = mean_squared_error(y_test, predictions, squared=False)
print(rmse)

#%% prepare data to compare
df_compare = pd.DataFrame({'Date':df_original['Date'][train_set_len + validation_set_len:], 'Actual Open':df_original['Open'][train_set_len + validation_set_len:]})
df_compare['Predicted Open'] = predictions

#%% visualize Actual Open, Predicted Open
plt.figure(figsize=(16,8))
plt.title('Actual Open vs. Predicted Open')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.plot(df_compare['Date'], df_compare['Actual Open'])
plt.plot(df_compare['Date'], df_compare['Predicted Open'])
plt.legend(['Actual','Predicted'], loc='upper right')
plt.show()

#%% visualize All Actual Open, Predicted Open
plt.figure(figsize=(16,8))
plt.title('All Actual Open vs. Predicted Open')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.plot(df_original['Date'], df_original['Open'])
plt.plot(df_compare['Date'], df_compare['Predicted Open'])
plt.legend(['Actual','Predicted'], loc='upper right')
plt.show()

#%%
# =============================================================================
# Hyperparameter Tuning (not used)
# =============================================================================
   
#%%
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        
        model = Sequential()        
        model.add(Input(shape=(x_train.shape[1],x_train.shape[2])))
        
        num_layers = hp.Int('num_rnn_layers', min_value=1, max_value=3, step=1, default=2)
        for i in range(num_layers):
            if i == num_layers - 1:
                model.add(LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=32), return_sequences=False))
                if hp.Boolean(f'lstm_{i}_dropout'):
                    model.add(Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
            else:
                model.add(LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=32), return_sequences=True))
                if hp.Boolean(f'lstm_{i}_dropout'):
                    model.add(Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
               
        model.add(Dense(units=hp.Int('dense_bef_out_units', min_value=8, max_value=32, step=8)))
        if hp.Boolean('dense_bef_out_dropout'):
            model.add(Dropout(rate=hp.Float('dense_bef_out_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Dense(units=1))
        
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
        
        # model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[16,32,64,128]),
            **kwargs,
        )

#%%
# =============================================================================
# BayesianOptimization Tuner (not used)
# =============================================================================

#%%
tuner = BayesianOptimization(
    MyHyperModel(),
    objective="val_loss",
    max_trials=10,
    overwrite=True,
    directory="bayes_dir",
    project_name="bayes_hypermodel",
)

#%%
tuner.search_space_summary()

#%%
early_stopped = EarlyStopping(monitor='val_loss', patience=5)

#%%
tuner.search(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), callbacks=[early_stopped])

#%% Get the top 2 models
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None,x_train.shape[1], x_train.shape[2]))
best_model.summary()

#%%
# =============================================================================
# Hyperband Tuner (not used)
# =============================================================================

#%%
tuner = Hyperband(
    MyHyperModel(),
    objective="val_loss",    
    max_epochs=100,    
    overwrite=True,
    directory="hyper_dir",
    project_name="hyper_hypermodel",
)

#%%
tuner.search_space_summary()

#%%
early_stopped = EarlyStopping(monitor='val_loss', patience=5)

#%%
tuner.search(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=[early_stopped])

#%% Get the top 2 models
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None,x_train.shape[1], x_train.shape[2]))
best_model.summary()


