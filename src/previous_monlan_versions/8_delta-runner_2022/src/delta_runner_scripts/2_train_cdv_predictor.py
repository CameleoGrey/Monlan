import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor

from src.monlan.utils.save_load import *
from src.delta_runner.ModelLSTM import ModelLSTM


dataset = load(os.path.join( "../../data", "dataset.pkl" ))
x_train, y_train = dataset["train"][0], dataset["train"][1]
x_test, y_test = dataset["test"][0], dataset["test"][1]
x_val, y_val = dataset["val"][0], dataset["val"][1]

#model = ModelLSTM(num_classes=1, input_size=128, hidden_size=128, num_layers=2, use_device="cpu")
#model.fit( x_train, y_train, x_val, y_val, epochs=20, learning_rate=1e-3 )

model = CatBoostRegressor(n_estimators=10000,
                          max_depth=8,
                          #thread_count=8,
                          task_type="GPU",
                          loss_function="MAPE",
                          early_stopping_rounds=200)
model.fit( x_train, y_train, eval_set=(x_val, y_val) )

#model = Ridge(alpha=1.0)
#model = ExtraTreesRegressor(n_estimators=400, max_depth=10, n_jobs=8 )
#model.fit( x_train, y_train )

save( model, os.path.join( "../../models", "cdv_predictor.pkl" ) )

model = load( os.path.join( "../../models", "cdv_predictor.pkl" ) )
y_pred = model.predict( x_test ) #* 4.0 #+ 0.3

scaler = load(os.path.join( "../../data", "scaler.pkl" ) )
y_pred = scaler.inverse_transform( y_pred.reshape((-1, 1)) )
y_test = scaler.inverse_transform( y_test.reshape((-1, 1)) )
#y_pred = np.exp( y_pred ) - 1.0
#y_test = np.exp( y_test ) - 1.0

train_pred = model.predict( x_train ) #* 4.0 #+ 0.3
train_pred = scaler.inverse_transform( train_pred.reshape((-1, 1)) )
low_volatility_mask = np.abs(y_pred) < 2.0 * np.std(train_pred)
save( np.std(train_pred), os.path.join( "../../models", "base_threshold.pkl" ) )

y_pred[ low_volatility_mask ] = np.mean(y_test)

x = [i for i in range(len(y_test))]
plt.plot( x, y_test, c="green" )
plt.plot( x, y_pred, c="orange" )
plt.show()

print("done")