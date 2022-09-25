# -*- coding: utf-8 -*-
"""

@author: tanap

dataset: https://finance.yahoo.com/quote/GE/history/

"""
#%%
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, CuDNNLSTM, Input, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner
from keras_tuner.tuners import BayesianOptimization

#%%
df = pd.read_csv('Dataset/GE.csv')

#%%
df_original = df
df_original['Date'] = pd.to_datetime(df_original['Date'])

#%%
plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.plot(df_original['Date'], df_original['Open'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.show()

#%%
# =============================================================================
# prepare data to split into Train set, Validation set, Test set
# =============================================================================

#%%
data = df.filter(['Open','High','Low','Close','Adj Close'])
dataset = data.values

#%%
train_set_len = math.ceil(len(dataset) * 0.80)
validation_set_len = math.ceil(len(dataset) * 0.10)

#%%
scaler = StandardScaler()
scaled_dataset = scaler.fit_transform(dataset)

#%%
n_future = 1
n_past = 14

#%%
# =============================================================================
# Train set
# =============================================================================

#%%
train_set = scaled_dataset[0:train_set_len, :]

x_train = []
y_train = []

for i in range(n_past, len(train_set) - n_future +1):
    x_train.append(train_set[i - n_past:i, 0:dataset.shape[1]])
    y_train.append(train_set[i + n_future - 1:i + n_future, 0])
    
#%%
x_train, y_train = np.array(x_train), np.array(y_train)

#%%
if x_train.ndim == 2:
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], 1))
    
#%%
# =============================================================================
# Validation set
# =============================================================================

#%%
validation_set = scaled_dataset[train_set_len - n_past:train_set_len + validation_set_len, :]

x_valid = []
y_valid = []

for i in range(n_past, len(validation_set) - n_future +1):
    x_valid.append(validation_set[i - n_past:i, 0:dataset.shape[1]])
    y_valid.append(validation_set[i + n_future - 1:i + n_future, 0])
    
#%%
x_valid, y_valid = np.array(x_valid), np.array(y_valid)

#%%
if x_valid.ndim == 2:
    x_valid = np.reshape(x_valid, (x_valid.shape[0],x_valid.shape[1], 1))
    
#%%
# =============================================================================
# Test set
# =============================================================================    

#%%
test_set = scaled_dataset[(train_set_len + validation_set_len) - n_past:, :]

x_test = []
y_test = dataset[train_set_len + validation_set_len:, 0]

for i in range(n_past, len(test_set)):
    x_test.append(test_set[i-n_past:i,0:dataset.shape[1]])
    
#%%
x_test = np.array(x_test)

#%%
if x_test.ndim == 2:
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

#%%
plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.plot(df_original.loc[0:train_set_len - 1, 'Date'],df_original.loc[0:train_set_len - 1, 'Open'])
plt.plot(df_original.loc[train_set_len - n_past:train_set_len + validation_set_len - 1, 'Date'],df_original.loc[train_set_len - n_past:train_set_len + validation_set_len - 1, 'Open'])
plt.plot(df_original.loc[(train_set_len + validation_set_len) - n_past:, 'Date'],df_original.loc[(train_set_len + validation_set_len) - n_past:, 'Open'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.legend(['Train set','Validation set','Test set'], loc='upper right')
plt.show()

#%%
# =============================================================================
# Hyperparameter Tuning
# =============================================================================

#%%
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = Sequential()
        
        model.add(Input(shape=(x_train.shape[1],x_train.shape[2])))
        
        for i in range(hp.Int('num_rnn_layers', min_value=1, max_value=3, step=1)):
            model.add(LSTM(units=hp.Int(f'lstm_t_{i}_units', min_value=32, max_value=128, step=16), return_sequences=True))
            if hp.Boolean(f'lstm_t_{i}_dropout'):
                model.add(Dropout(rate=hp.Float(f'lstm_t_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(LSTM(units=hp.Int('lstm_f_units', min_value=32, max_value=128, step=16), return_sequences=False))
        if hp.Boolean('lstm_f_dropout'):
            model.add(Dropout(rate=hp.Float('lstm_f_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Dense(units=hp.Int('dense_bef_out_units', min_value=8, max_value=32, step=4)))
        if hp.Boolean('dense_bef_out_dropout'):
            model.add(Dropout(rate=hp.Float('dense_bef_out_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Dense(units=1))
        
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse', metrics=['mse'])
        
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[8,16,32,64]),
            **kwargs,
        )

#%%
tuner = BayesianOptimization(
    MyHyperModel(),
    objective="val_loss",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

#%%
tuner.search_space_summary()

#%%
early_stopped = EarlyStopping(monitor='val_loss', patience=5)

#%%
tuner.search(x_train, y_train, epochs=100, validation_data=(x_valid, y_valid), callbacks=[early_stopped])

#%%
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None,x_train.shape[1], x_train.shape[2]))
best_model.summary()

#%%
# =============================================================================
# Build Model
# =============================================================================

#%%
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32))
model.add(Dense(1))

#%%
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.summary()

#%%
early_stopped = EarlyStopping(monitor='val_loss', patience=5)

#%%
checkpoint_filepath = 'checkpoint'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch',
    verbose=1
    )

#%%
history1 = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), callbacks=[model_checkpoint_callback, early_stopped])

#%%
plt.figure(figsize=(16,8))
plt.title('Loss rate')
plt.plot(history1.history['loss'], label='Training loss')
plt.plot(history1.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(['Training loss','Validation loss'], loc='upper right')
plt.show()

#%%
predictions = model.predict(x_test)
predictions_repeated = np.repeat(predictions, dataset.shape[1], axis=-1)
predictions_final = scaler.inverse_transform(predictions_repeated)[:,0]

#%%
rmse = np.sqrt(np.mean(predictions_final-y_test)**2)
print(rmse)

#%%
df_forecast = df_original.loc[train_set_len + validation_set_len: , ['Date']]
df_forecast['Predicted Open'] = predictions_final

# df_original = df_original.loc[train_set_len + validation_set_len: , :]

#%%
plt.figure(figsize=(16,8))
plt.title('Model')
plt.plot(df_original['Date'], df_original['Open'])
plt.plot(df_forecast['Date'], df_forecast['Predicted Open'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price USD ($)', fontsize=18)
plt.legend(['Actual','Predicted'], loc='upper right')
plt.show()

#%%
df_forecast['Actual Open'] = y_test
df_forecast
