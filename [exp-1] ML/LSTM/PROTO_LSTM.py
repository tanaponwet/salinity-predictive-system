# -*- coding: utf-8 -*-
"""
@author: tanap

dataset: https://finance.yahoo.com/quote/GE/history/
"""
#%%
import numpy as np
import pandas as pd

#%%
df_read = pd.read_csv('data/GE.csv')

#%%
df = df_read.copy()
print(df.dtypes)

#%%
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)

#%%
import matplotlib.pyplot as plt

plt.figure(figsize=(16,8))
plt.title('Open Price History')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.plot(df['Date'], df['Open'])
plt.show()

#%%
data = df.filter(['Open','High','Low','Close','Adj Close'])
dataset = data.values

#%%
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
dataset_scaled = scaler.fit_transform(dataset)

#%%
n_future = 1
n_past = 14

#%%
import math

train_set_len = math.ceil(len(dataset) * 0.80)
validation_set_len = math.ceil(len(dataset) * 0.10)

#%%
train_start = 0
train_end = train_set_len
valid_start = train_set_len - n_past
valid_end = train_set_len + validation_set_len
test_start = (train_set_len + validation_set_len) - n_past
test_end = len(dataset_scaled)
y_test_start = train_set_len + validation_set_len
y_test_end = len(dataset_scaled)

#%%
train_set = dataset_scaled[train_start:train_end, :]
validation_set = dataset_scaled[valid_start:valid_end, :]
test_set = dataset_scaled[test_start:test_end, :]

#%%
def to_sequences(dataset, n_past=14, n_future=1, shape=1):
    x = []
    y = []
    
    for i in range(n_past, len(dataset) - n_future +1):
        x.append(dataset[i - n_past:i, 0:shape])
        y.append(dataset[i + n_future - 1:i + n_future, 0])
    
    return np.array(x), np.array(y)

#%%
x_train, y_train = to_sequences(train_set, n_past=n_past, n_future=n_future, shape=dataset.shape[1])
x_valid, y_valid = to_sequences(validation_set, n_past=n_past, n_future=n_future, shape=dataset.shape[1])
x_test, y_dummy = to_sequences(test_set, n_past=n_past, n_future=n_future, shape=dataset.shape[1])

y_test = dataset[y_test_start:y_test_end, 0]

#%%
plt.figure(figsize=(16,8))
plt.title('Train, Valid, Test Visualization')
plt.xlabel('Data')
plt.ylabel('Open Price USD ($)')
plt.plot(df[train_start:train_end]['Date'], df[train_start:train_end]['Open'])
plt.plot(df[valid_start:valid_end]['Date'], df[valid_start:valid_end]['Open'])
plt.plot(df[test_start:test_end]['Date'], df[test_start:test_end]['Open'])
plt.show()

#%%
# =============================================================================
# Build Model
# =============================================================================

#%%
from keras import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import SimpleRNN
from keras.layers import LSTM
# from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
# from tensorflow.keras.layers import Bidirectional
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

#%% RNN
model = Sequential()
model.add(SimpleRNN(64, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(SimpleRNN(64, return_sequences=False))
# model.add(Dense(16))
model.add(Dense(1))

#%% LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(LSTM(64, return_sequences=False))
# model.add(Dense(16))
model.add(Dense(1))

#%% Bidirectional LSTM
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
# model.add(Dense(16))
model.add(Dense(1))

#%%
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

#%%
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#%%
checkpoint_filepath = 'checkpoint/my_model.h5'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch',
    verbose=1
    )

#%%
model_fit = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), callbacks=[model_checkpoint_callback, early_stopping_callback])

#%%
plt.figure(figsize=(16,8))
plt.title('Loss rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(model_fit.history['loss'], label='Training loss')
plt.plot(model_fit.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#%%
pred = model.predict(x_test)

#%%
if dataset.shape[1] > 1:
    pred = np.repeat(pred, dataset.shape[1], axis=-1)

#%%
prediction = scaler.inverse_transform(pred)[:,0]

#%%
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, prediction, squared=False)
print(rmse)

#%%
df_compare = pd.DataFrame({'Date':df['Date'][y_test_start:], 'Actual Open':df['Open'][y_test_start:]})
df_compare['Predicted Open'] = prediction

#%%
plt.figure(figsize=(16,8))
plt.title('Actual Open vs. Predicted Open')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.plot(df_compare['Date'], df_compare['Actual Open'], label='Actual')
plt.plot(df_compare['Date'], df_compare['Predicted Open'], label='Predicted')
plt.legend()
plt.show()

#%%
plt.figure(figsize=(16,8))
plt.title('All Actual Open vs. Predicted Open')
plt.xlabel('Date')
plt.ylabel('Open Price USD ($)')
plt.plot(df['Date'], df['Open'], label='Actual')
plt.plot(df_compare['Date'], df_compare['Predicted Open'], label='Predicted')
plt.legend()
plt.show()

#%%
# =============================================================================
# Hyperparameter Tuning (not used)
# =============================================================================

#%%
import keras_tuner
from keras_tuner.tuners import BayesianOptimization, Hyperband

#%%
# def build_model(hp):
    
#     inputs = Input(shape=(14,5))
#     x = inputs
    
#     num_layers = hp.Int('num_rnn_layers', min_value=1, max_value=3, step=1)
#     for i in range(num_layers):
#         if i == num_layers - 1:
#             x = LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=False)(x)
#             if hp.Boolean(f'lstm_{i}_dropout'):
#                 x = Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
#         else:
#             x = LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=True)(x)
#             if hp.Boolean(f'lstm_{i}_dropout'):
#                 x = Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
    
#     if hp.Boolean('dense_bef_out'):
#         x = Dense(units=hp.Int('dense_bef_out_units', min_value=8, max_value=32, step=4))(x)
#         if hp.Boolean('dense_bef_out_dropout'):
#             x = Dropout(rate=hp.Float('dense_bef_out_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
            
#     outputs = Dense(units=1)(x)
    
#     model = Model(inputs, outputs)
    
#     hp_optimizer = hp.Choice("optimizer",values = ["adam","RMSprop","SGD"])
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
#     optimizer = None
#     if hp_optimizer == "adam":
#         optimizer = Adam(learning_rate = hp_learning_rate)
#     elif hp_optimizer == "RMSprop":
#         optimizer = RMSprop(learning_rate = hp_learning_rate)
#     elif hp_optimizer == "SGD":
#         optimizer = SGD(learning_rate = hp_learning_rate)
    
#     model.compile(optimizer=optimizer,loss='mse',metrics = ['mse'])
    
#     return model

#%%
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        
        model = Sequential()        
        model.add(Input(shape=(14,5)))
        
        num_layers = hp.Int('num_rnn_layers', min_value=1, max_value=3, step=1)
        for i in range(num_layers):
            if i == num_layers - 1:
                model.add(LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=False))
                if hp.Boolean(f'lstm_{i}_dropout'):
                    model.add(Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
            else:
                model.add(LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=True))
                if hp.Boolean(f'lstm_{i}_dropout'):
                    model.add(Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
               
        if hp.Boolean('dense_bef_out'):
            model.add(Dense(units=hp.Int('dense_bef_out_units', min_value=8, max_value=32, step=4)))
            if hp.Boolean('dense_bef_out_dropout'):
                model.add(Dropout(rate=hp.Float('dense_bef_out_dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Dense(units=1))
        
        hp_optimizer = hp.Choice("optimizer",values = ["adam","RMSprop","SGD"])
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        optimizer = None
        if hp_optimizer == "adam":
            optimizer = Adam(learning_rate = hp_learning_rate)
        elif hp_optimizer == "RMSprop":
            optimizer = RMSprop(learning_rate = hp_learning_rate)
        elif hp_optimizer == "SGD":
            optimizer = SGD(learning_rate = hp_learning_rate)
        
        model.compile(optimizer=optimizer,loss='mse',metrics = ['mse'])
        
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', values=[32,64,128]),
            **kwargs,
        )
    
#%%
# class MyHyperModel(keras_tuner.HyperModel):
#     def build(self, hp):
        
#         inputs = Input(shape=(14,5))
#         x = inputs
        
#         num_layers = hp.Int('num_rnn_layers', min_value=1, max_value=3, step=1)
#         for i in range(num_layers):
#             if i == num_layers - 1:
#                 x = LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=False)(x)
#                 if hp.Boolean(f'lstm_{i}_dropout'):
#                     x = Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
#             else:
#                 x = LSTM(units=hp.Int(f'lstm_{i}_units', min_value=32, max_value=128, step=16), return_sequences=True)(x)
#                 if hp.Boolean(f'lstm_{i}_dropout'):
#                     x = Dropout(rate=hp.Float(f'lstm_{i}_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
        
#         if hp.Boolean('dense_bef_out'):
#             x = Dense(units=hp.Int('dense_bef_out_units', min_value=8, max_value=32, step=4))(x)
#             if hp.Boolean('dense_bef_out_dropout'):
#                 x = Dropout(rate=hp.Float('dense_bef_out_dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
                
#         outputs = Dense(units=1)(x)
        
#         model = Model(inputs, outputs)
        
#         hp_optimizer = hp.Choice("optimizer",values = ["adam","RMSprop","SGD"])
#         hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
#         optimizer = None
#         if hp_optimizer == "adam":
#             optimizer = Adam(learning_rate = hp_learning_rate)
#         elif hp_optimizer == "RMSprop":
#             optimizer = RMSprop(learning_rate = hp_learning_rate)
#         elif hp_optimizer == "SGD":
#             optimizer = SGD(learning_rate = hp_learning_rate)
        
#         model.compile(optimizer=optimizer,loss='mse',metrics = ['mse'])
        
#         return model

#     def fit(self, hp, model, *args, **kwargs):
#         return model.fit(
#             *args,
#             batch_size=hp.Choice('batch_size', values=[32,64,128]),
#             **kwargs,
#         )

#%%
# =============================================================================
# BayesianOptimization (not used)
# =============================================================================

#%%
tuner = BayesianOptimization(
    # build_model,
    MyHyperModel(),
    objective="val_loss",
    max_trials=50,
    overwrite=True,
    directory="bayes_dir",
    project_name="bayes_hypermodel",
)

#%%
tuner.search_space_summary()

#%%
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#%%
tuner.search(
    x_train,
    y_train,
    # batch_size=64,
    epochs=64,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping_callback]
    )

#%%
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None,x_train.shape[1], x_train.shape[2]))
best_model.summary()

#%%
# =============================================================================
# Hyperband (not used)
# =============================================================================

#%%
tuner = Hyperband(
    # build_model,
    MyHyperModel(),
    objective="val_loss",    
    max_epochs=64,    
    overwrite=True,
    directory="hyband_dir",
    project_name="hyband_hypermodel",
)

#%%
tuner.search_space_summary()

#%%
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

#%%
tuner.search(
    x_train,
    y_train,
    # batch_size=64,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stopping_callback]
    )

#%%
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.build(input_shape=(None,x_train.shape[1], x_train.shape[2]))
best_model.summary()

