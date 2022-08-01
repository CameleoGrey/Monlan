
from monlan.envs.CompositeEnv import CompositeEnv
from monlan.envs.RealCompositeEnv import RealCompositeEnv
from monlan.feature_generators.W2VCompositeGenerator import W2VCompositeGenerator
from monlan.feature_generators.W2VScaleGenerator import W2VScaleGenerator
from monlan_supervised.MultiScalerDiffGenerator import MultiScalerDiffGenerator
from monlan.datamanagement.SymbolDataManager import SymbolDataManager
from monlan.datamanagement.SymbolDataUpdater import SymbolDataUpdater
from monlan.terminal.MT5Terminal import MT5Terminal
from monlan.mods.HeikenAshiMod import HeikenAshiMod
from monlan.mods.EnergyMod import EnergyMod
from monlan.mods.VSASpread import VSASpread
import matplotlib.pyplot as plt
from datetime import datetime
from monlan_supervised.SamplesGenerator import SamplesGenerator
import numpy as np
from monlan.utils.save_load import *
import gc
from tqdm import tqdm

from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_percentage_error
from keras.models import Model, Input
from keras.layers import Dense, Dropout, BatchNormalization, Reshape, Flatten
from keras.optimizers import Adam
from keras.models import model_from_json, load_model, Sequential
from keras.regularizers import l2, l1_l2
from catboost import CatBoostRegressor

class LSTMPredictor():
    def __init__(self):
        self.model = None
        pass

    def buildModel(self, inputDim, lr=0.001):

        regressor = Sequential()
        # First LSTM layer with Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(inputDim, 1)))
        regressor.add(Dropout(0.2))
        # Second LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Third LSTM layer
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Fourth LSTM layer
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        # The output layer
        regressor.add(Dense(units=1))

        self.model = regressor

        self.model.compile(optimizer=Adam(lr), loss="mse")
        return self

    def fit(self, x, y, batchSize, epochs):
        self.model.fit(x, y, batch_size=batchSize, epochs=epochs)
        return self


startTime = datetime.now()
print("start time: {}".format(startTime))

#symbols = ["EURUSD_i"]
symbols = ["AUDUSD_i", "EURAUD_i", "USDCAD_i", "EURCAD_i", "GBPUSD_i",]
timeframe = "M10"


dataManager = SymbolDataManager("../data/raw/")

df = dataManager.getData(symbols[0], timeframe)
df = df.tail(400000)
x =  df["open"] - df["open"].shift(1)
#x = x.values[1:].reshape(-1,1)
x = x.values[1:]
#x = df["open"].values
#scaler = StandardScaler().fit(x.reshape((-1, 1)))
scaler = MinMaxScaler().fit(x.reshape((-1, 1)))
x = scaler.transform(x.reshape(-1,1))

window = 64
x_train, y_train = [], []
for i in tqdm(range(len(x) - window), colour="green"):
    x_train.append( x[i:i+window] )
    y_train.append( x[i+window] )
x_train = np.array(x_train)
y_train = np.array(y_train)
#x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
#y_train = y_train.reshape((y_train.shape[0], y_train.shape[1]))

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
x_train_shuffled, y_train_shuffled = shuffle(x_train, y_train, random_state=45)

#model = CatBoostRegressor(n_estimators=1000, max_depth=4, early_stopping_rounds=200, verbose=1)
#model.fit(x_train_shuffled, y_train_shuffled, eval_set=(x_test, y_test))

model = LSTMPredictor().buildModel(x_train_shuffled[0].shape[0])
model.fit(x_train_shuffled, y_train_shuffled, batchSize=32, epochs=10)

y_pred = model.predict(x_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print("MAPE = {:.2%}".format(mape))

x = [i for i in range(len(y_test))]
plt.plot(x, y_test)
plt.plot(x, y_pred)
plt.show()

